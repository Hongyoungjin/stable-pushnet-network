#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author : Sungwon Seo(ssw0536@g.skku.edu)
Date   : 2023-01-22
"""
import rospy
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseStamped


class CameraTransformBroadcaster(object):
    def __init__(self):
        # Load transformations
        self.tf_sbr = StaticTransformBroadcaster()


    def get_m1013_camera_transform(self, camera_pose_msg):
        assert isinstance(camera_pose_msg, PoseStamped)
        position = camera_pose_msg.pose.position
        orientation = camera_pose_msg.pose.orientation
        # get m1013 camera transform
        m1013_cam_tf_msg = TransformStamped()
        m1013_cam_tf_msg.header.frame_id = 'm1013_base_0'
        m1013_cam_tf_msg.header.stamp = rospy.Time.now()
        m1013_cam_tf_msg.child_frame_id = 'm1013_camera'
        m1013_cam_tf_msg.transform.translation.x = position.x
        m1013_cam_tf_msg.transform.translation.y = position.y
        m1013_cam_tf_msg.transform.translation.z = position.z
        m1013_cam_tf_msg.transform.rotation.x = orientation.x
        m1013_cam_tf_msg.transform.rotation.y = orientation.y
        m1013_cam_tf_msg.transform.rotation.z = orientation.z
        m1013_cam_tf_msg.transform.rotation.w = orientation.w
        return m1013_cam_tf_msg

    def broadcast_transforms(self, camera_pose_msg):
        try:
            m1013_cam_tf_msg = self.get_m1013_camera_transform(camera_pose_msg)
            self.tf_sbr.sendTransform([m1013_cam_tf_msg])
            rospy.sleep(0.1)
            rospy.loginfo('Broadcasted transforms')
            return 0
        except NotImplementedError as e:
            rospy.logwarn('Failed to broadcast transforms: {}'.format(e))


if __name__ == '__main__':
    # init ros node
    rospy.init_node('camera_trasform_visualizer')

    # init visualizer
    visualizer = CameraTransformBroadcaster()

    # broadcast transforms
    visualizer.broadcast_transforms()
    rospy.spin()
