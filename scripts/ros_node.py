#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import collision
import matplotlib
matplotlib.use('TkAgg')

import rospy
import tf
import tf.transformations as tft
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf.transformations as tft
from copy import copy

from stable_pushnet_ros.srv import GetStablePushPath, GetStablePushPathRequest
from stable_pushnet_ros.msg import PushTarget
from stable_pushing.stable_push_planner import HybridAstarPushPlanner
from stable_pushing.stable_determinator import DepthImageBasedDeterminator, StablePushNetDeterminator
from stable_pushing.contact_point_sampler import ContactPointSampler
from stable_pushing.map_interface import MapInterface

    
class StablePushNetServer(object):
    
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.tf = tf.TransformerROS()
        self.map_interface = MapInterface()
        

        self.planner_config = rospy.get_param("~planner")
        self.hybrid_config = rospy.get_param("~hybrid")
        self.depth_based_config = rospy.get_param("~depth_based")

        rospy.loginfo("planner_config: {}".format(self.planner_config))
        rospy.loginfo("hybrid_config: {}".format(self.hybrid_config))
        rospy.loginfo("depth_based_config: {}".format(self.depth_based_config))

        # initialize ros service
        rospy.Service(
            '~get_stable_push_path',
            GetStablePushPath,
            self.get_stable_push_path_handler)

        # Get stable push path
            
        stable_determinator = DepthImageBasedDeterminator(
            self.depth_based_config['friction_coefficient'])
        
        if self.planner_config['learning_base'] == True:
            stable_determinator = StablePushNetDeterminator()
            
        self.planner = HybridAstarPushPlanner(
            stable_determinator=stable_determinator,
            grid_size=self.hybrid_config['grid_size'],
            dtheta=np.radians(self.hybrid_config['dtheta']))

        # print
        rospy.loginfo('StablePushNetServer is ready to serve.')

    def get_stable_push_path_handler(self, request):
        assert isinstance(request, GetStablePushPathRequest)
        # ros msgs
        dish_seg_msg = request.dish_segmentation
        table_det_msg = request.table_detection
        depth_img_msg = request.depth_image
        camera_info_msg = request.cam_info
        camera_pose_msg = request.camera_pose
        push_target_array_msg = request.push_targets
        
        rospy.loginfo("Received request.")
        
        # Parse push target msg
        target_id_list, goal_pose_list, push_direction_range_list = self.parse_push_target_msg(push_target_array_msg)
        
        # Get visual information 
        cam_pos_tran = [camera_pose_msg.pose.position.x, camera_pose_msg.pose.position.y, camera_pose_msg.pose.position.z]
        cam_pos_quat = [camera_pose_msg.pose.orientation.x, camera_pose_msg.pose.orientation.y, camera_pose_msg.pose.orientation.z, camera_pose_msg.pose.orientation.w]
        cam_pos = self.tf.fromTranslationRotation(cam_pos_tran, cam_pos_quat)
        
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
        cam_intr = np.array(camera_info_msg.K).reshape(3, 3)

        # Get dish information 
        segmask_list, id_list = self.parse_dish_segmentation_msg(dish_seg_msg)
        dish_shape_list = self.map_interface.get_dish_shapes(id_list, segmask_list, depth_img, cam_pos, cam_intr)
        rospy.loginfo("{} Dishes are segmented".format(len(dish_shape_list)))
        
        # Get table information
        map_corners = self.parse_table_detection_msg(table_det_msg) # min_x, max_x, min_y, max_y
        
        # Loop through push path planning until we find the push path
        for i in range(len(target_id_list)):
            # Get current target ID
            target_id = target_id_list[i]
            goal_pose = goal_pose_list[i]
            push_direction_range = push_direction_range_list[i]
            
            # Get corresponding data
            segmask_img = segmask_list[target_id]
            
            # Sample push contacts
            cps = ContactPointSampler(cam_intr, cam_pos, 
                                    gripper_width = self.planner_config['gripper_width'],
                                    num_push_dirs = self.planner_config['num_push_directions'],
                                    push_dir_range = push_direction_range)
            contact_points = cps.sample(depth_img, segmask_img)
            
            # Set map obstacles
            obstacles = copy(dish_shape_list)
            obstacles.pop(target_id)
            map_obstacles = self.shrink_obstacles_for_initial_collision(contact_points, obstacles)

            # set goal        
            goal = goal_pose

            # Derive stable push path
            self.planner.update_map(map_corners, map_obstacles)
            
            if self.planner_config['image_type'] == 'masked':
                input_image = np.multiply(depth_img, segmask_img)
            else: 
                input_image = np.nan_to_num(depth_img)
                
            best_path, plan_successful = self.planner.plan(
                input_image, contact_points, goal, #depth_img * segmask_img is for masked depth image
                learning_base=self.planner_config['learning_base'],
                visualize=self.planner_config['visualize'])

            # Make ros msg
            path_msg = Path()
            path_msg.header.frame_id = camera_pose_msg.header.frame_id
            path_msg.header.stamp = rospy.Time.now()
            
            if not plan_successful:
                rospy.logwarn("Planning failed.")
                
                if i != len(target_id_list) - 1:
                    continue
                
                best_path = [[1,1,10], [10,1,2]]
            
            for each_point in best_path:
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = rospy.Time.now()
                pose_stamped.header.frame_id = camera_pose_msg.header.frame_id
                pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z = each_point[0], each_point[1], self.planner_config['height']
                pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w = tft.quaternion_from_euler(each_point[2],0,0-np.pi, axes='rzyx')
                path_msg.poses.append(pose_stamped)

            return path_msg, plan_successful
        
    def parse_push_target_msg(self, push_target_array_msg):
        ''' Parse push target array msg to target ids and push directions.'''
        priority_list = []
        target_id_list = []
        goal_pose_list = []
        push_direction_range_list = []
        
        
        for target in push_target_array_msg.push_targets:
            assert isinstance(target, PushTarget)
            priority_list.append(target.priority)
            target_id_list.append(target.push_target_id)
            goal_pose_list.append([target.goal_pose.x, target.goal_pose.y, target.goal_pose.theta])
            push_direction_range_list.append([target.start_pose_min_theta.theta, target.start_pose_max_theta.theta])
            
        priority_array = np.array(priority_list)
        target_id_array = np.array(target_id_list)
        goal_pose_array = np.array(goal_pose_list)
        push_direction_range_array = np.array(push_direction_range_list)
        
        # Sort target goals by priority
        sorted_indices = np.argsort(priority_array)
        
        target_id_list = target_id_array[sorted_indices]
        goal_pose_list = goal_pose_array[sorted_indices]
        push_direction_range_list = push_direction_range_array[sorted_indices]
            
        return target_id_list, goal_pose_list, push_direction_range_list
    
    
    def parse_dish_segmentation_msg(self, dish_segmentation_msg):
        ''' Parse dish segmentation msg to segmasks and ids.'''
        
        segmasks = []
        
        for detection in dish_segmentation_msg.detections:
            # Get segmask
            segmask_msg = detection.source_img
            segmask = self.cv_bridge.imgmsg_to_cv2(segmask_msg, desired_encoding='passthrough')
            segmasks.append(segmask)
            
        # Since all dishes have same segmentation id, we have to manually assign ids
        ids = [i for i in range(len(segmasks))]
        
        return segmasks, ids
        
    def parse_table_detection_msg(self, table_det_msg):
        ''' Parse table detection msg to table pose.'''
        
        position_msg = table_det_msg.center.position
        orientation_msg = table_det_msg.center.orientation
        size_msg = table_det_msg.size
        
        position = np.array([position_msg.x, position_msg.y, position_msg.z])
        orientation = np.array([orientation_msg.x, orientation_msg.y, orientation_msg.z, orientation_msg.w])
        
        rot_mat = tft.quaternion_matrix(orientation)[:3,:3]
        
        # Get local positions of vertices 
        vertices_loc = []
        for x in [-size_msg.x/2, size_msg.x/2]:
            for y in [-size_msg.y/2, size_msg.y/2]:
                for z in [-size_msg.z/2, size_msg.z/2]:
                    vertices_loc.append([x,y,z])
        vertices_loc = np.array(vertices_loc)
        
        # Convert to world frame
        vertices_world = np.matmul(rot_mat, vertices_loc.T).T + position
        
        x_max, x_min = np.max(vertices_world[:,0]), np.min(vertices_world[:,0])
        y_max, y_min = np.max(vertices_world[:,1]), np.min(vertices_world[:,1])
        # z_max, z_min = np.max(vertices_world[:,2]), np.min(vertices_world[:,2])
            
        return [x_min, x_max, y_min, y_max]
    
    @staticmethod
    def visualize_map(map_obstacles, contact_points):
        contact_point = contact_points[0]
        min_bbox = contact_point.min_bbox
        pose = contact_point.pose
        
        # Get the bounding_box in the world frame
        rot_mat = np.array([[np.cos(pose[2]), -np.sin(pose[2])],
                            [np.sin(pose[2]),  np.cos(pose[2])]])
        min_bbox = np.matmul(rot_mat, min_bbox.T).T + pose[:2]
        
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        min_bbox = np.vstack((min_bbox, min_bbox[0]))
        ax.plot(min_bbox[:,0], min_bbox[:,1], c='r')
        for obstacle in map_obstacles:
            obstacle_points = []
            print(len(obstacle.points))
            for point in obstacle.points:
                obstacle_points.append([point.x, point.y])
            obstacle_points = np.array(obstacle_points)
            print(obstacle_points.shape)
            ax.plot(obstacle_points[:,0], obstacle_points[:,1], c='b')
        ax.set_aspect('equal')  
            
        plt.show()
        
    def shrink_obstacles_for_initial_collision(self, contact_points, obstacles):
        """Resolve initial collision between the slider and the obstacles.
        
        If the slider is initially in collision with the obstacles,
        the obstacle polygon is reduced to avoid the collision.

        Args:
            obstacles (List[points (float)]): List of 2D vertices of obstacles.
            contact_points (List[ContactPoint]): List of contact points.

        Returns:
            reduced_obstacles (List[collision.Polygon]): List of reduced collision polygons.
        """
        
        contact_point = contact_points[0]
        min_bbox = contact_point.min_bbox
        pose = contact_point.pose
        
        # Get the bounding_box in the world frame
        rot_mat = np.array([[np.cos(pose[2]), -np.sin(pose[2])],
                            [np.sin(pose[2]),  np.cos(pose[2])]])
        min_bbox = np.matmul(rot_mat, min_bbox.T).T + pose[:2]
        
        reduced_obstacles = []
        
        for obstacle in obstacles:
            
            # Reduce the obstacle shape to avoid collision
            reduced_obstacle_points = self.map_interface.reduce_overlap(min_bbox, obstacle, buffer_distance=0.07)
            reduced_obstacle_points = np.array(reduced_obstacle_points)
            
            # Modify the obstacle object to reduced shape 
            reduced_obstacle = collision.Poly(collision.Vector(0,0),
                                          [collision.Vector(point[0], point[1]) for point in reduced_obstacle_points])
            
            reduced_obstacles.append(reduced_obstacle)
        
        return reduced_obstacles
    

if __name__ == '__main__':
    rospy.init_node('stable_pushnet_server')
    server = StablePushNetServer()
    
    rospy.spin()


