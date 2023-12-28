import numpy as np
import cv2
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import trimesh
from .utils.minimum_bounding_box import MinimumBoundingBox
import alphashape
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
import shapely.geometry as geometry
import math
from shapely.ops import cascaded_union, polygonize
from shapely import Polygon, MultiPolygon


class ContactPoint(object):
    def __init__(self, edge_xyz, edge_uv, contact_points, contact_points_uv, push_direction):
        """Push contact point.
        Args:
            edge_xyz (numpy.ndarray): (N, 3) array of edge points in world frame.
            edge_uv (numpy.ndarray): (N, 2) array of edge points in image coordinates.
            contact_points (numpy.ndarray): (N, 2, 3) contact points in world frame.
            contact_points_uv (numpy.ndarray): (N, 2,2) contact points in image coordinates.
            push_direction (numpy.ndarray): (2,) array of pushing  direction in world frame.
        """
        
        self.edge_xyz = edge_xyz
        self.edge_uv = edge_uv
        self.contact_points = contact_points
        self.contact_points_uv = contact_points_uv
        self.push_direction = push_direction

    @property
    def contact_normals(self):
        """Set contact normals.
        Args:
            contact_normals (numpy.ndarray): (2,2) contact normals in world frame.
        """
        ch = ConvexHull(self.edge_xyz[:, :2])
        ch_xy = self.edge_xyz[ch.vertices][:, :2]
        surface_normal1 = self._get_surface_normal(ch_xy, self.contact_points[0])
        surface_normal2 = self._get_surface_normal(ch_xy, self.contact_points[1])
        return np.array([surface_normal1, surface_normal2])

    @property
    def pose(self):
        """Position in world frame.
        Returns:
            pose (numpy.ndarray): Position (x, y theta) in world frame.
        """
        position = self.contact_points.mean(0)
        orientation = np.arctan2(self.push_direction[1], self.push_direction[0])
        return np.array([position[0], position[1], orientation])

    @staticmethod
    def _get_surface_normal(convex_hull, contact_point):
        """
        Get a surface normal from convex hull and a contact point.
        
        Args:
            convex_hull (numpy.ndarray): shape (N, 2)
            contact_point (numpy.ndarray): shape (3, )
            
        Returns:
            surface_normal (numpy.ndarray): shape (2, )
        """
        dist = np.linalg.norm(contact_point[:2] - convex_hull, axis=-1)
        candidate_index = np.argsort(dist)[:2]
        
        if np.abs(candidate_index[0] - candidate_index[1]) < (len(convex_hull) - 1):
            edge_point1 = convex_hull[np.min(candidate_index)]
            edge_point2 = convex_hull[np.max(candidate_index)]
        else:
            edge_point1 = convex_hull[np.max(candidate_index)]
            edge_point2 = convex_hull[np.min(candidate_index)]

        surface_normal = edge_point2 - edge_point1
        surface_normal = surface_normal / np.linalg.norm(surface_normal)
        surface_normal = np.array([-surface_normal[1], surface_normal[0]])
        return surface_normal
    
    def visualize_on_image(self, image):
        contact_points_uv = self.contact_points_uv.reshape(-1,2)
        contact_point1 = contact_points_uv[0]
        contact_point2 = contact_points_uv[1]
        center = np.mean(contact_points_uv, axis=0)
        
        push_direcion = contact_point2 - contact_point1
        push_direcion = np.array([push_direcion[1], -push_direcion[0]], dtype=np.float64)
        push_direcion /= np.linalg.norm(push_direcion)

        # visualize contact point
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap='gray')
        ax.scatter(
            contact_point1[0],
            contact_point1[1], c='g', marker='o')
        ax.scatter(
            contact_point2[0],
            contact_point2[1],
            c='r', marker='o')
        ax.arrow(
            center[0], center[1],
            push_direcion[0] * 50,
            push_direcion[1] * 50,
            color='b', head_width=10, head_length=10)
        plt.show()

    def visualize_on_cartesian(self):
        # get random 1000 point on edge_xy
        rand_idx = np.random.randint(0, len(self.edge_xyz), 1000)
        
        fig = plt.figure()
        # plot edge points
        plt.plot(self.edge_xyz[rand_idx, 0], self.edge_xyz[rand_idx, 1], 'ko')
        # plot contact points
        plt.plot(self.contact_points[0, 0], self.contact_points[0, 1], 'go')
        plt.plot(self.contact_points[1, 0], self.contact_points[1, 1], 'ro')
        # plot push direction
        plt.arrow(
            self.pose[0], self.pose[1],
            self.push_direction[0] * 0.05, self.push_direction[1] * 0.05,
            color='b', head_width=0.01, head_length=0.01)
        # plot surface normal
        plt.arrow(
            self.contact_points[0, 0], self.contact_points[0, 1],
            self.contact_normals[0, 0] * 0.05, self.contact_normals[0, 1] * 0.05,
            color='r', head_width=0.01, head_length=0.01)   
        plt.arrow(
            self.contact_points[1, 0], self.contact_points[1, 1],
            self.contact_normals[1, 0] * 0.05, self.contact_normals[1, 1] * 0.05,
            color='r', head_width=0.01, head_length=0.01)   
        plt.axis('equal')
        plt.show()   
        
    @property
    def min_bbox(self):
        """
        Minimum bounding box in contact local frame.

        Returns:
            min_bbox (numpy.ndarray): (4, 2) minimum bounding box.
        """
        min_bbox = MinimumBoundingBox(self.edge_xyz[:,:2]).corner_points
        min_bbox = np.array(list(min_bbox))
        
        # transfrom min bbox to local frame
        min_bbox = min_bbox - self.pose[:2]
        rot_mat = np.array([
            [np.cos(-self.pose[2]), -np.sin(-self.pose[2])],
            [np.sin(-self.pose[2]), np.cos(-self.pose[2])]])
        min_bbox = np.dot(rot_mat, min_bbox.T)
        min_bbox = min_bbox.T

        # sort as ccw
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

        def convex_hull(points):
            n = len(points)
            l = 0
            for i in range(1,n):
                if points[i][0] < points[l][0]:
                    l = i
            hull = [points[l]]
            p = l
            while True:
                q = (p+1)%n
                for i in range(n):
                    if ccw(points[p], points[i], points[q]):
                        q = i
                p = q
                if p == l:
                    break
                hull.append(points[p])
            return np.array(hull)

        return convex_hull(min_bbox)
        
class ContactPointSampler(object):
    '''
    Samples contact points from a depth image
    Depth image -> Contact points 
    
    '''
    def __init__(self, camera_intr, camera_extr, gripper_width=0.08, num_push_dirs=8, push_dir_range=[-np.pi/2, np.pi/2]):
        self.camera_intr = camera_intr
        self.camera_extr = camera_extr
        self.gripper_width = gripper_width
        self.num_push_dirs = num_push_dirs
        self._width_error_threshold = 1e-5
        
        self.push_dir_range = push_dir_range
        
    def sample(self, depth_image, mask):
        edge_list_uv, edge_list_xyz = self.edge_list_using_pcd(depth_image, mask, self.camera_extr, self.camera_intr)
        contact_pair_uv, contact_pair_xyz = self.get_contact_points(edge_list_uv, edge_list_xyz)
        edge_center = edge_list_xyz.mean(0)
        contact_pair_centers = contact_pair_xyz.mean(1)
        
        # Calculate pushing directions
        pushing_directions = contact_pair_xyz[:,0] - contact_pair_xyz[:,1]
        pushing_directions = pushing_directions[:,:2]
        pushing_directions = np.roll(pushing_directions, 1, axis=1)
        pushing_directions[:,1] *= -1
        
        edge_list_xy = edge_list_xyz[:, :2]
        contact_pair_centers_xy = contact_pair_centers[:, :2]
        
        
        projections_contact_point_centers = np.einsum('ij,ij->i', pushing_directions, contact_pair_centers_xy).reshape(-1, 1)
        projections_edges = np.einsum('ij,kj -> ik', pushing_directions, edge_list_xy)
        projections_edges_min_max = np.vstack([projections_edges.min(1), projections_edges.max(1)]).T
        projections_edges_median = projections_edges_min_max.mean(1).reshape(-1,1)
        
        pushing_directions = np.where(projections_edges_median > projections_contact_point_centers, pushing_directions, -pushing_directions)
                
        contact_pair_angles = np.arctan2(pushing_directions[:,1], pushing_directions[:,0])
        
        # min_theta is smaller than max_theta (regular case)
        sorted_indices_regular = np.where(np.logical_and(contact_pair_angles[0] <= contact_pair_angles[1],
                                                            np.logical_and(self.push_dir_range[0] <= contact_pair_angles,
                                                                          contact_pair_angles <= self.push_dir_range[1])))[0]
        
        # min_theta < pi & -pi < max_theta (exceptional case - due to the periodic characteristic of radian notation)
        sorted_indices_exceptional = np.where(np.logical_and(contact_pair_angles[0] > contact_pair_angles[1],
                                                            np.logical_or(self.push_dir_range[0] <= contact_pair_angles,
                                                                          contact_pair_angles <= self.push_dir_range[1])))[0]
        sorted_indices = np.concatenate((sorted_indices_regular, sorted_indices_exceptional))
        
        # sorted_indices = np.argsort(contact_pair_angles)
        
        step = sorted_indices.shape[0] // self.num_push_dirs

        try:
            sorted_indices = sorted_indices[::step]
        except:
            print(f'Error: number of contact points is less than {self.num_push_dirs}')
            pass
            
        contact_points = []
        
        for idx in sorted_indices:
            
            contact_points.append(ContactPoint(edge_list_xyz, edge_list_uv, contact_pair_xyz[idx], contact_pair_uv[idx], pushing_directions[idx]))
            
        return contact_points
    
    def get_contact_points(self, edge_list_uv, edge_list_xyz):
        '''
        Sample contact points from depth contour
        Args:
            edge_list_xy (N,2): Edge indices list in uv coordinates 
            edge_list_xy (N,2): Edge list in xy coordinates (world frame)
            
        Returns:
            contact_uv_coordinates: (N,2,2): Sampled contact points in depth image pixel format ([(v11,u11),(v12,u12)], [(v21,u21),(v22,u22)] ... [(vN1,uN1),(vN2,uN2)])
            contact_pair_pcd:       (N,2,3): Sampled contact points in 3D coordiantes {world} ([(x11,y11,z11),(x12,y12,z12)], [(x21,y21,z21),(x22,y22,z22)] ... [(xN1,yN1,zN1),(xN2,yN2,zN2)])
        
        '''
        edge_list_xy = edge_list_xyz[:,:2]
        distances = cdist(edge_list_xy, edge_list_xy)
        # Find the point index pairs of cetrain euclidean distance
        contact_pair_idx = np.where(np.abs(distances - self.gripper_width) <= self._width_error_threshold)
        contact_pair_idx = np.vstack((contact_pair_idx[0],contact_pair_idx[1])).T
        contact_pair_idx = np.unique(contact_pair_idx, axis=0) # remove duplicates
        
        contact_pair_xyz = np.hstack((edge_list_xyz[contact_pair_idx[:,0]], edge_list_xyz[contact_pair_idx[:,1]])).reshape(-1,2,3)
        contact_pair_uv = np.hstack((edge_list_uv[contact_pair_idx[:,0]], edge_list_uv[contact_pair_idx[:,1]])).reshape(-1,2,2)
        
        return contact_pair_uv, contact_pair_xyz
    
    @staticmethod
    def remove_outliers(array, threshold=3):
        # Calculate the mean and standard deviation of the array
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)

        # Calculate the Z-scores for each data point
        z_scores = np.abs((array - mean) / std)

        # Filter out the outliers based on the threshold
        filtered_array = array[(z_scores < threshold).all(axis=1)]

        return filtered_array

    def edge_list_using_pcd(self, depth_image, segmask, camera_extr, camera_intrinsic):
        '''
        Reproject depth image in vertical view
        '''
        
        # Get point cloud of the object only
        depth_image = depth_image * segmask
        pcd = self.depth_to_pcd(depth_image, camera_intrinsic)
        pcd_object = pcd[np.where(pcd[:,2] > 0.1)[0]]
        
        # Transform point cloud to world frame
        pcd_w = (np.matmul(camera_extr[:3,:3], pcd_object[:,:3].T) + camera_extr[:3,3].reshape(3,1)).T
        
        #########################
        #  Height Thresholding ##
        #########################
        
        threshold_height = 0.01
        # Remove points that are too close to the ground
        pcd_w = pcd_w[np.where(pcd_w[:,2] > threshold_height)[0]]
        
        ##########################################
        # Edge Detection - alpha shape algorithm #
        ##########################################
        
        # Calculate the Delaunay triangulation of the point cloud
        pcd_w_2d = pcd_w[:,:2]

        # Define the alpha value (adjust according to your data)
        # alpha_value = 500

        # # Calculate the alpha shape of the point cloud
        # alpha_shape = alphashape.alphashape(pcd_w_2d, alpha=alpha_value)
        
        # if type(alpha_shape) == MultiPolygon:
        #     xs, ys = [], []
        #     for poly in alpha_shape.geoms:
        #         exterior = poly.exterior.coords[:]
        #         x, y = zip(*exterior)
        #         xs += list(x)
        #         ys += list(y)
        #     xs = np.array(xs).reshape(-1, 1)
        #     ys = np.array(ys).reshape(-1, 1)
        #     outermost_points = np.hstack((xs, ys))
        # elif type(alpha_shape) == Polygon:
        #     outermost_points = np.array(alpha_shape.exterior.coords)
        # Get the points on the precise contour
        # outermost_points = np.array(alpha_shape.exterior.coords)
        
        # Find the convex hull of the point cloud
        hull = ConvexHull(pcd_w_2d)

        # Get the indices of the points on the outermost contour
        outermost_indices = hull.vertices
        
        # Get the points on the outermost contour
        outermost_points = pcd_w_2d[outermost_indices]
        
        # Extract x and y coordinates from the contour points
        x = outermost_points[:, 0]
        y = outermost_points[:, 1]
        num_interpolated_points = 500
        # Create an interpolation function for x and y coordinates separately
        interpolation_function_x = interp1d(np.arange(len(x)), x, kind='linear')
        interpolation_function_y = interp1d(np.arange(len(y)), y, kind='linear')

        # Generate evenly spaced indices for interpolation
        interpolation_indices = np.linspace(0, len(x)-1, num=num_interpolated_points)

        # Interpolate x and y coordinates using the interpolation functions
        x_interpolated = interpolation_function_x(interpolation_indices)
        y_interpolated = interpolation_function_y(interpolation_indices)

        # Create the interpolated trajectory with m points (m, 2)
        interpolated_contour_points = np.column_stack((x_interpolated, y_interpolated))
        edge_list_xyz = np.hstack([interpolated_contour_points, 0.005 * np.zeros(len(interpolated_contour_points)).reshape(-1,1)]).reshape(-1,3)
        
        # Get uv coordinates of the edge list
        edge_list_xyz_camera = (np.matmul(np.linalg.inv(camera_extr)[:3,:3], edge_list_xyz[:,:3].T) + np.linalg.inv(camera_extr)[:3,3].reshape(3,1)).T
        edge_list_uvd = edge_list_xyz_camera @ camera_intrinsic.T
        edge_list_uv = edge_list_uvd[:,:2] / edge_list_uvd[:,2].reshape(-1,1)
        edge_list_uv = edge_list_uv.astype(int)
        
        return edge_list_uv, edge_list_xyz

    @staticmethod
    def alpha_shape(points, alpha):
        """
        Compute the alpha shape (concave hull) of a set of points.

        @param points: Iterable container of points.
        @param alpha: alpha value to influence the gooeyness of the border. Smaller
                    numbers don't fall inward as much as larger numbers. Too large,
                    and you lose everything!
        """
        if len(points) < 4:
            # When you have a triangle, there is no sense in computing an alpha
            # shape.
            return geometry.MultiPoint(list(points)).convex_hull

        def add_edge(edges, edge_points, coords, i, j):
            """Add a line between the i-th and j-th points, if not in the list already"""
            if (i, j) in edges or (j, i) in edges:
                # already added
                return
            edges.add( (i, j) )
            edge_points.append(coords[ [i, j] ])

        # coords = np.array([point.coords[0] for point in points])

        tri = Delaunay(points)
        edges = set()
        edge_points = []
        # loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.vertices:
            pa = coords[ia]
            pb = coords[ib]
            pc = coords[ic]

            # Lengths of sides of triangle
            a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
            b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
            c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

            # Semiperimeter of triangle
            s = (a + b + c)/2.0

            # Area of triangle by Heron's formula
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
            circum_r = a*b*c/(4.0*area)

            # Here's the radius filter.
            #print circum_r
            if circum_r < 1.0/alpha:
                add_edge(edges, edge_points, coords, ia, ib)
                add_edge(edges, edge_points, coords, ib, ic)
                add_edge(edges, edge_points, coords, ic, ia)

        m = geometry.MultiLineString(edge_points)
        triangles = list(polygonize(m))
        return cascaded_union(triangles), edge_points
    @staticmethod
    def pcd_idx_to_uv_coordinates(contact_pair_idx, depth_img_shape):
        """Get xy pixel coordinates from point cloud indices
        Args:
            contact_pair_idx (N,2): PCD indices of contact pairs ([idx_11,idx_12], [idx_21,idx_22] ... [idx_N1,idx_N2])
        Returns:
            (N,(2,2)): Contact points in depth image pixel format ([(v_11,u_11),(v_12,u_12)], [(v_21,u_21),(v_22,u_22)] ... [(v_N1,u_N1),(v_N2,u_N2)])
            
            Note: 
            1. the order of the contact points is not guaranteed to be the same as the input
            2. Returned coordinates start with v, not u. (v,u) = (x,y)
            
        """
        first_indices, second_indices = contact_pair_idx[:,0], contact_pair_idx[:,1]
        H, W = depth_img_shape[0], depth_img_shape[1]
        y1, x1= np.unravel_index(first_indices,  (H,W))
        y2, x2= np.unravel_index(second_indices, (H,W))
        contact_uv_coordinates = np.hstack((y1.reshape(-1,1), x1.reshape(-1,1), y2.reshape(-1,1), x2.reshape(-1,1))).reshape(-1,2,2)
        
        return contact_uv_coordinates
    
    @staticmethod
    def depth_to_pcd(depth_image, camera_intr):
        height, width = depth_image.shape
        row_indices = np.arange(height)
        col_indices = np.arange(width)
        pixel_grid = np.meshgrid(col_indices, row_indices)
        pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
        pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
        depth_arr = np.tile(depth_image.flatten(), [3, 1])
        point_cloud = depth_arr * np.linalg.inv(camera_intr).dot(pixels_homog)
        return point_cloud.transpose()