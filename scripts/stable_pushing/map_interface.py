import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import parmap
import multiprocessing
from itertools import repeat
from std_msgs.msg import Float32
from shapely.geometry import Polygon


class MapInterface(object):
    
    def __init__(self, debug=False):
        
        self.num_cores = multiprocessing.cpu_count()
        
    def get_dish_shapes(self, id_list, segmask_list, depth_image, camera_pose, camera_intrinsic):
        '''Get dish shapes from input data'''
        
        shape_with_id_list = parmap.starmap(self.detect_edges_xy, 
                                list(zip(id_list, segmask_list, repeat(depth_image), repeat(camera_pose), repeat(camera_intrinsic))),
                                pm_processes=self.num_cores, pm_chunksize = self.num_cores)
        
        shape_with_id_list = sorted(shape_with_id_list, key=lambda x: list(x.keys())[0])
        shape_list_xy = [list(shape_dict.values())[0] for shape_dict in shape_with_id_list]
        
        return shape_list_xy
    
    def get_table_info(self, segmask, depth_image, camera_pose, camera_intrinsic):
        '''Get table information from input data'''
        
        table_edges = self.detect_edges_xy(segmask, depth_image, camera_pose, camera_intrinsic)
        edges_x, edges_y = table_edges[0], table_edges[1]
        
        max_x, min_x = np.max(edges_x), np.min(edges_x)
        max_y, min_y = np.max(edges_y), np.min(edges_y)
        
        table_info= [Float32(max_x), Float32(min_x), Float32(max_y), Float32(min_y)]
        
        return table_info
    
    def get_table_info_tmp(self, segmask, depth_image, camera_pose, camera_intrinsic):
        # [max_x, min_x, max_y, min_y]
        table_info= [Float32(-0.2), Float32(-1.050), Float32(0.3875), Float32(-0.4125)]
        table_info= [0.1, -1.050, 0.3875, -0.4125]
        
        return table_info
    
    def detect_edges_xy(self, id, segmask, depth_image, camera_pose, camera_intrinsic):
        '''
        Get 2D shape of the object in the image
        '''
        
        # Get point cloud of the object only
        depth_image = depth_image * segmask
        pcd = self.depth_to_pcd(depth_image, camera_intrinsic)
        pcd_object = pcd[np.where(pcd[:,2] > 0.1)[0]]
        
        # Transform point cloud to world frame
        pcd_w = (np.matmul(camera_pose[:3,:3], pcd_object[:,:3].T) + camera_pose[:3,3].reshape(3,1)).T
        
        #########################
        #  Height Thresholding ##
        #########################
        
        threshold_height = 0.01
        # Remove points that are too close to the ground
        pcd_w = pcd_w[np.where(pcd_w[:,2] > threshold_height)[0]]
        
        ##################
        # Edge Detection #
        ##################
        
        # Calculate the Delaunay triangulation of the point cloud
        pcd_w_2d = pcd_w[:,:2]
        
        # Find the convex hull of the point cloud
        hull = ConvexHull(pcd_w_2d)

        # Get the indices of the points on the outermost contour
        outermost_indices = hull.vertices
        
        # Get the points on the outermost contour
        outermost_points = pcd_w_2d[outermost_indices]
        
        # Extract x and y coordinates from the contour points
        x = outermost_points[:, 0]
        y = outermost_points[:, 1]
        num_interpolated_points = 20
        
        # Create an interpolation function for x and y coordinates separately
        interpolation_function_x = interp1d(np.arange(len(x)), x, kind='linear')
        interpolation_function_y = interp1d(np.arange(len(y)), y, kind='linear')

        # Generate evenly spaced indices for interpolation
        interpolation_indices = np.linspace(0, len(x)-1, num=num_interpolated_points)

        # Interpolate x and y coordinates using the interpolation functions
        x_interpolated = interpolation_function_x(interpolation_indices)
        y_interpolated = interpolation_function_y(interpolation_indices)

        # Create the interpolated trajectory with m points (m, 2)
        edge_list_xy = np.column_stack((x_interpolated, y_interpolated))
        
        return {id: edge_list_xy}
    
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
    
    @staticmethod
    def _compute_overlap(polygon_A, polygon_B):
        """
        Computes the overlapping region of two polygons.
        
        Args:
        - polygon_A (list of tuples): List of (x, y) coordinates for polygon A.
        - polygon_B (list of tuples): List of (x, y) coordinates for polygon B.
        
        Returns:
        - Overlapping region as a Shapely geometry.
        """
        poly_A = Polygon(polygon_A)
        poly_B = Polygon(polygon_B)
        return poly_A.intersection(poly_B)
    
    @staticmethod
    def reduce_overlap(polygon_A, polygon_B, buffer_distance=0.01):
        '''
        Reduces the overlapping region of two polygons.
        
        Args:
        - polygon_A (list of tuples): List of (x, y) coordinates for polygon A.
        - polygon_B (list of tuples): List of (x, y) coordinates for polygon B. Polygon_B is reduced.
        - buffer_distance (float): Distance to buffer the overlapping region by.

        Returns
        - Reduced polygon_B as a list of (x, y) coordinates.'''
        overlap = MapInterface._compute_overlap(polygon_A, polygon_B)
        buffered_overlap = overlap.buffer(buffer_distance)
        reduced_B = Polygon(polygon_B).difference(buffered_overlap)
        
        if isinstance(reduced_B, Polygon):
            reduced_B_coords = list(reduced_B.exterior.coords)
        else:
            all_coords = [list(poly.exterior.coords) for poly in reduced_B]
            reduced_B_coords = [coord for coords in all_coords for coord in coords]
        
        return reduced_B_coords
    
    @staticmethod
    def add_overlap(polygon_A, polygon_B):
        '''
        Adds the overlapping region of two polygons.
        
        Args:
        - polygon_A (list of tuples): List of (x, y) coordinates for polygon A.
        - polygon_B (list of tuples): List of (x, y) coordinates for polygon B. Polygon_B is Added.
        - buffer_distance (float): Distance to buffer the overlapping region by.

        Returns
        - Added polygon_B as a list of (x, y) coordinates.'''
        
        
        overlap = MapInterface._compute_overlap(polygon_A, polygon_B)
        merged_poly = Polygon(polygon_A).union(overlap)
        
        if isinstance(merged_poly, Polygon):
            merged_coords = list(merged_poly.exterior.coords)
        else:
            all_coords = [list(poly.exterior.coords) for poly in merged_poly]
            merged_coords = [coord for coords in all_coords for coord in coords]
        
        return merged_coords