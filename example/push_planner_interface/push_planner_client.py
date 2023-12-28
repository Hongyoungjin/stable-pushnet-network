#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author : Sungwon Seo(ssw0536@g.skku.edu)
Date   : 2023-01-22
"""
import rospy
import stable_pushnet_ros.srv as stable_pushnet_ros_srv


class StablePushnetPlannerClient(object):
    def __init__(self):
        # register service
        service_name = '/stable_push_planner/get_stable_push_path'
        rospy.wait_for_service(service_name)
        self.get_stable_push_path = rospy.ServiceProxy(
            service_name, stable_pushnet_ros_srv.GetStablePushPath())
        rospy.loginfo("Service `%s` is ready" % service_name)

    def reqeust(self, dish_seg_msg, table_det_msg, depth_image_msg, depth_cam_info_msg, camera_pose_msg, push_targets):
        """Request stable push path from min_icr_push_planner.

        Parameters
        ----------
        depth_image : sensor_msgs/Image
            Depth image.
        segmask : sensor_msgs/Image
            Segmentation mask.
        camera_info : sensor_msgs/CameraInfo
            Camera info.
        camera_pose : geometry_msgs/PoseStamped
            Camera pose in robot base frame.
        map_info : MapInfo
            Map info msg.
        goal_pose : geometry_msgs/PoseStamped
            Goal pose in robot base frame.
        push_direction_range : [float, float]
            Push direction range in radian.

        Returns
        -------
        push_path : nav_msgs/Path
            Push path.
        push_contact : list of ContactPoint
            Push contact points.
        """
        rospy.loginfo("Requesting stable push path")
        start_time = rospy.Time.now()
        try:
            req = stable_pushnet_ros_srv.GetStablePushPathRequest()
            req.dish_segmentation = dish_seg_msg
            req.table_detection = table_det_msg
            req.depth_image = depth_image_msg
            req.cam_info = depth_cam_info_msg
            req.camera_pose = camera_pose_msg
            req.push_targets = push_targets
            res = self.get_stable_push_path(req)
            rospy.loginfo('Service call succeeded. Elapsed time: {}'.format(
                (rospy.Time.now() - start_time).to_sec()))
            return res.path, res.plan_successful
        
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            rospy.logwarn("Service call failed")