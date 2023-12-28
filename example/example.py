#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import seaborn as sns

import rospy
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, Image

# Push path module service client
from push_planner_interface.push_planner_client import StablePushnetPlannerClient

# For vizualization
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from robot_camera_calibration.camera_transform_broadcaster import CameraTransformBroadcaster

# For loading example data
import pickle


# python 2/3 compatibility
try:
    input = raw_input
except NameError:
    print('python 2/3 compatiblity: input() is changed to raw_input(). Delete this try-except block when you update to noetic.')
    pass

class StablePushTask(object):
    def __init__(self):
        
        # Initialize planner module
        self.init_push_planner()
        
        # Get pre-made example data
        self.get_example_data()

        # Load camera transform and broadcast to /tf
        self.cam_tf_broadcaster = CameraTransformBroadcaster()
        self.cam_tf_broadcaster.broadcast_transforms(self.camera_pose_msg)

        self.cv_bridge = CvBridge()

        # publish for visualization
        self.push_path_pub = rospy.Publisher('/push_path', Path, queue_size=1)
        self.point_cloud_pub = rospy.Publisher('/point_cloud2', PointCloud2, queue_size=2)
        self.color_segmask_pub = rospy.Publisher('/color_segmask', Image, queue_size=2)
    
    @staticmethod
    def get_color_mask_image(color_image, segmask_image):
        """get segment mask labeled color image

        Args:
            color_image (`numpy.ndarray`): (H, W, C) color image.
            segmask_image (`numpy.ndarray`): (H, W, class) segment mask image.
        """
        color_palette = np.array(sns.color_palette("Dark2", 20))

        # copy color image
        img = color_image.copy()
        temp_color_palette = (color_palette * 255).astype("uint8")

        # label segment mask on the color image
        for i in range(int(np.max(segmask_image)) + 1):
            if i == 0:
                continue
            color_mask = np.zeros_like(color_image, dtype=np.uint8)
            color_mask[segmask_image == i] = temp_color_palette[i]
            img = cv2.addWeighted(img, 1, color_mask, 1, 0)

        return img
        
    @staticmethod
    def show_example_segmented_scene(scene_img):
        # Visualize segmentation result
        plt.figure()
        plt.imshow(scene_img)
        plt.show()
    
    def init_push_planner(self):
        self.push_planner_client = StablePushnetPlannerClient()
        rospy.loginfo('Push planner initialized.')
        
    def get_example_data(self):
        
        def open_pickle(filename):
            file_path = os.path.join(os.path.dirname(__file__), "example_data", filename)
            with open(file_path, 'rb') as f:
                return pickle.load(f)
            
        self.dish_seg_msg = open_pickle('dish_seg_msg')
        self.table_det_msg = open_pickle('table_det_msg')
        self.scene_img = open_pickle('scene_img')
        self.camera_pose_msg = open_pickle('camera_pose')
        self.depth_image_msg = open_pickle('depth_image_msg')
        self.point_cloud2_msg = open_pickle('point_cloud2_msg')
        self.segmask = open_pickle('segmask')
        self.color_img = open_pickle('color_img')
        self.depth_cam_info_msg = open_pickle('depth_cam_info_msg')
        self.push_target_array_msg = open_pickle('push_target_array_msg')

    def visualize_example_scene_in_rviz(self):
        
        '''Visualize point cloud & color segmask in rViz'''
        self.point_cloud2_msg.header.frame_id = 'm1013_camera'
        self.point_cloud_pub.publish(self.point_cloud2_msg)
        color_segmask = self.get_color_mask_image(self.color_img, self.segmask)
        color_segmask_msg = self.cv_bridge.cv2_to_imgmsg(color_segmask, "rgb8")
        self.color_segmask_pub.publish(color_segmask_msg)
        
    def request_push_path(self):
        
        # Visualize example scene (does not affect planning)
        self.show_example_segmented_scene(self.scene_img)
        self.visualize_example_scene_in_rviz()
        
        # Request push planning
        push_path, plan_successful = self.push_planner_client.reqeust(self.dish_seg_msg, 
                                                                      self.table_det_msg, 
                                                                      self.depth_image_msg, 
                                                                      self.depth_cam_info_msg, 
                                                                      self.camera_pose_msg, 
                                                                      self.push_target_array_msg)
        
        if not plan_successful:
            rospy.logerr('Push planning Failed.')
            return
        
        # Visualize planned push path in rViz
        self.push_path_pub.publish(push_path)

if __name__ == '__main__':
    rospy.init_node('stable_push_task')
    stable_push_task = StablePushTask()

    while True:
        user_input = input('Press enter to start task, q to quit...')
        if user_input == 'q':
            break
        elif user_input == '':
            pass
        else:
            continue

        stable_push_task.request_push_path()
