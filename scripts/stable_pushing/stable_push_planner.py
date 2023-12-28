#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import datetime
import copy
from dataclasses import dataclass
import yaml
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import collision

from corgipath.collision import BoundingVolumeHierarchy
from corgipath.search_space import DefaultHybridGrid, DefaultHybridNode, HybridSuccessor
from corgipath.planning import HybridAstar
from corgipath.matplot import static_draw as draw
from corgipath.matplot import live_draw as live
from corgipath.matplot.utils import pick_color, auto_scale

from .utils.hybrid_astar import HybridAstarTrajectory
from .utils.utils import crop_image
from .contact_point_sampler import ContactPoint, ContactPointSampler
from .stable_determinator import StablePushNetDeterminator, DepthImageBasedDeterminator
from scipy.interpolate import CubicSpline

from typing import Tuple, Dict

@dataclass
class PushHybridNode(DefaultHybridNode):
    _cached_centroid = None

    @property
    def centroid(self) -> Tuple[float, float, float]:
        if self._cached_centroid is None:
            # Calculate dxdy from center
            dx = np.cos(self.xyt[2]) * self._centroid_offset[0] - np.sin(self.xyt[2]) * self._centroid_offset[1]
            dy = np.sin(self.xyt[2]) * self._centroid_offset[0] + np.cos(self.xyt[2]) * self._centroid_offset[1]
            self._cached_centroid = (
                self.xyt[0] + dx,
                self.xyt[1] + dy,
                self.xyt[2])
        return self._cached_centroid


class HybridAstarPushPlanner(object):
    def __init__(self, stable_determinator, grid_size, dtheta):
        """Initialize the HybridAstarPushPlanner.

        Args:
            stable_determinator (StageDeterminator): The stage determinator object.
            grid_size (float): The grid size in meters.
            dtheta (float): The direction resolution in rad.
        """
        
        # parameters
        self._grid_size = grid_size
        self._dtheta = dtheta
        self._corners = None
        self._obstacles = None

        # objects        
        self._stable_determinator = stable_determinator
        self._planner = HybridAstar()    

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
    
    def _get_stable_minimum_radius(self, depth_image, contact_point):
        """Get the stable minimum ICRs from a given depth image and contact point.

        Note:
            The ICR is along the y-axis.

        Args:
            depth_image (numpy.ndarray): Depth image in (H, W).
            contact_point (ContactPoint): Contact point in world coordinate.

        Returns:
            left_min_radius (float): The minimum radius( > 0) in meters.
            right_min_radius (float): The minimum radius( > 0) in meters.
        """
        assert isinstance(self._stable_determinator, StablePushNetDeterminator)
        log_radius = np.linspace(np.log10(1e-4), np.log10(100), 200)
        radius_positive = np.power(10, log_radius)
        radius = np.concatenate((np.flip(-radius_positive), radius_positive))
        icrs = np.vstack((np.zeros_like(radius), radius)).T
        results = self._stable_determinator.get_push_results(depth_image, contact_point, icrs)
        
        left_results, right_results = np.flip(results[:200]), results[200:]
        # left_radius_valid  = self.remove_outliers(radius_positive[left_results  == 1].reshape(-1,1))
        # right_radius_valid = self.remove_outliers(radius_positive[right_results == 1].reshape(-1,1))
        left_radius_valid  = radius_positive[left_results  == 1].reshape(-1,1)
        right_radius_valid = radius_positive[right_results == 1].reshape(-1,1)
        
        try:
            left_min_radius = np.min(left_radius_valid)
        except:
            left_min_radius = 100
        try:
            right_min_radius = np.min(right_radius_valid)
        except:
            right_min_radius = 100
            
        print('left_min_radius: {}, right_min_radius: {}'.format(
            left_min_radius, right_min_radius))
        
        # safety_factor = 1.5
        # left_min_radius, right_min_radius = left_min_radius * safety_factor, right_min_radius * safety_factor
        
        return left_min_radius, right_min_radius
    
    def _get_stable_minimum_radius_depth_base(self, depth_image, contact_point):
        """Get the stable minimum ICRs from a given depth image and contact point.

        Note:
            The ICR is along the y-axis.

        Args:
            depth_image (numpy.ndarray): Depth image in (H, W).
            contact_point (ContactPoint): Contact point in world coordinate.

        Returns:
            left_min_radius (float): The minimum radius( > 0) in meters.
            right_min_radius (float): The minimum radius( > 0) in meters.
        """
        
        # Left(y > 0) - do binary search to find the minimum radius
        min_radius = 0
        max_radius = 100
        while abs(min_radius - max_radius) > 0.005:
            radius = np.random.uniform(min_radius, max_radius)
            is_stable = self._stable_determinator.is_stable(depth_image, contact_point, (0, radius))
            if is_stable:
                max_radius = radius
            else:
                min_radius = radius + 0.005
        left_min_radius = max_radius

        # Right(y < 0): do binary search to find the minimum radius
        min_radius = 0
        max_radius = 100
        while abs(min_radius - max_radius) > 0.005:
            radius = np.random.uniform(min_radius, max_radius)
            if self._stable_determinator.is_stable(depth_image, contact_point, (0, -radius)):
                max_radius = radius
            else:
                min_radius = radius + 0.005
        right_min_radius = max_radius
        
        print('left_min_radius: {}, right_min_radius: {}'.format(
            left_min_radius, right_min_radius))
        
        # safety_factor = 1.5
        # left_min_radius, right_min_radius = left_min_radius * safety_factor, right_min_radius * safety_factor
        
        return left_min_radius, right_min_radius

    def _compare_stability_for_all_velocities(self, depth_image, contact_points):
        analytic_determinator = DepthImageBasedDeterminator(friction_coefficient=0.5)
        trained_determinator = StablePushNetDeterminator()
        _velocity = trained_determinator.velocities
        
        fig = plt.figure()
        
        for idx, contact_point in enumerate(contact_points):
            
            # Show given image
            ax = fig.add_subplot(len(contact_points),3,3*idx+1)
            masked_image_mean = trained_determinator.masked_image_mean
            masked_image_std = trained_determinator.masked_image_std
            image_mean = trained_determinator.image_mean
            image_std = trained_determinator.image_std
            
            image = crop_image(depth_image, contact_point)
            
            # image = (image - image_mean) / image_std
            # image = (image - masked_image_mean) / masked_image_std
            ax.imshow(image)
            
            # Show reference result
            qualities = analytic_determinator.check_all_velocities(depth_image, contact_point)
            
            # ax = fig.add_subplot(9,3,3*idx+2,projection='3d')
            ax = fig.add_subplot(len(contact_points),3,3*idx+2)
            # ax.view_init(elev = 0,azim = 0)
            # ax.set_title(f"Analytical Result")
            # ax.set_xlabel(r"$v_x$ [m/s]")
            # ax.set_ylabel(r"$v_y$ [m/s]")
            # ax.set_zlabel(r"$\omega$ [rad]", rotation=0)
            # ax.set_box_aspect((1,2,2))
            ax.set_aspect('equal')
            ax.grid(False)
            # p = ax.scatter(_velocity[:,1], _velocity[:,2], qualities, c=qualities, cmap="jet", s=30)
            p = ax.scatter(_velocity[:,1], _velocity[:,2],  c=qualities, cmap="jet", s=15)
            
            # Show network result
            qualities = trained_determinator.check_all_velocities(depth_image, contact_point)
            # ax = fig.add_subplot(9,3,3*idx+3,projection='3d')
            ax = fig.add_subplot(len(contact_points),3,3*idx+3)
            # ax.view_init(elev = 0,azim = 0)
            # ax.set_title(f"Network Result")
            # ax.set_xlabel(r"$v_x$ [m/s]")
            # ax.set_ylabel(r"$v_y$ [m/s]")
            # ax.set_zlabel(r"$\omega$ [rad]", rotation=0)
            # ax.set_box_aspect((1,2,2))
            ax.set_aspect('equal')
            ax.grid(False)
            # p = ax.scatter( _velocity[:,1], _velocity[:,2], qualities, c=qualities, cmap="jet", s=30, vmin=0, vmax=1)
            p = ax.scatter( _velocity[:,1], _velocity[:,2],  c=qualities, cmap="jet", s=15, vmin=0, vmax=1)
            fig.colorbar(p, ax=ax)
        
        plt.show()
        
    def _get_successor_template(self, left_min_radius, right_min_radius):
        """Set the successor template to the default template.

        Args:
            left_min_radius (float): The minimum radius( > 0) in meters.
            right_min_radius (float): The minimum radius( > 0) in meters.
        """
        assert left_min_radius > 0 and right_min_radius > 0

        # In Hybrid A*, minimum distance to forward >= diagonal length.
        diagonal = np.sqrt(2.0) * self._grid_size

        # Pair of displacement and cost
        forward = (diagonal, diagonal)
        edges = (forward,)

        # Max heading change from min radius
        max_heading_change_left = diagonal / left_min_radius
        max_heading_change_right = diagonal / right_min_radius
        print('left_heading_angle: {}, right_heading_angle: {}'.format(
            np.rad2deg(max_heading_change_left), np.rad2deg(max_heading_change_right)))

        # limit max heading change
        if max_heading_change_left > np.pi:
            print('max_left_heading_angle > pi : {}. Limit to pi.'.format(
                max_heading_change_left))
            max_heading_change_left = np.pi
        if max_heading_change_right > np.pi:
            print('max_right_heading_angle > pi : {}. Limit to pi.'.format(
                max_heading_change_right))
            max_heading_change_right = np.pi

        # TODO: self._dtheta is changed inside function. Fix it.
        num_of_choices = 5
        self._dtheta = min(max_heading_change_left, max_heading_change_right) / float(num_of_choices)
        self._dtheta = np.pi/float(int(np.pi/self._dtheta) + 1)
        print('Set _dtheta: {}. Inside `_get_successor_template`.'.format(self._dtheta))

        # Make template
        template = []
        for displacement, cost in edges:
            template.append(HybridSuccessor.from_heading_with_dist(0.0, displacement, cost))
            _dist = displacement / np.sqrt(2.0)
            _cost = cost / np.sqrt(2.0)
            for i in range(1, int(max_heading_change_left / self._dtheta ) + 1):
                template.append(HybridSuccessor.from_heading_with_dist(self._dtheta * i, _dist, _cost))
            for i in range(1, int(max_heading_change_right / self._dtheta ) + 1):
                template.append(HybridSuccessor.from_heading_with_dist(-self._dtheta * i, _dist, _cost))
        return template

    def _get_collision_system(self, contact_point):
        """get the collision system.

        Args:
            contact_point (ContactPoint): Contact point.

        Returns:
            collision_system (BoundingVolumeHierarchy): Bounding volume hierarchy.
        """
        # Define static objects (obstacles) here.
        collision_system = BoundingVolumeHierarchy(bounds=self._corners)
        collision_system.add_obstacles(self._obstacles)
        # X-axis is forward in the agent's frame
        collision_system.agent_collision = collision.Poly(
            collision.Vector(0, 0),
            [collision.Vector(p[0], p[1]) for p in contact_point.min_bbox])
        return collision_system

    def _get_search_space(self, successor_template):
        """Define the search space.

        Args:
            icr_left (numpy.ndarray): [icr_x, icr_y] for left side.
            icr_right (numpy.ndarray): [icr_x, icr_y] for right side.
            
        Returns:
            search_space (DefaultHybridGrid): The search space.
        """
        # Define the search space
        search_space = DefaultHybridGrid(
            dxy=self._grid_size,
            dtheta=self._dtheta, node_type=PushHybridNode)
        search_space.successor_template = successor_template
        return search_space

    @staticmethod
    def _cartesian_terminal_condition(query: PushHybridNode, goal: PushHybridNode) -> bool:
        """Cartesian terminal condition for Hybrid A*.

        Args:
            query (PushHybridNode): Query node. It should be subclass of PushHybridNode.
            goal (PushHybridNode): Goal node. It should be subclass of PushHybridNode.

        Returns:
            bool: True if the query index is the same as the goal index.
        """
        qcentroid = np.array([query.centroid[0], query.centroid[1]])
        gcentroid = np.array([goal.centroid[0], goal.centroid[1]])
        return np.linalg.norm(qcentroid - gcentroid) < 0.02

    @staticmethod
    def _se2_terminal_condition(query: PushHybridNode, goal: PushHybridNode) -> bool:
        """Cartesian terminal condition for Hybrid A*.

        Args:
            query (PushHybridNode): Query node. It should be subclass of PushHybridNode.
            goal (PushHybridNode): Goal node. It should be subclass of PushHybridNode.

        Returns:
            bool: True if the query index is the same as the goal index.
        """
        raise NotImplementedError
        return query.index == goal.index

    @staticmethod
    def _cartesian_heuristic(query: PushHybridNode, goal: PushHybridNode) -> float:
        """Default heuristic function for Hybrid A*.

        This heuristic function assumes Z-shaped path to approximate a cubic Bézier curve.

        Args:
            query (PushHybridNode): Query node. It should be subclass of PushHybridNode.
            goal (PushHybridNode): Goal node. It should be subclass of PushHybridNode.

        Returns:
            float: Heuristic score from current node to goal
        """
        qx, qy, qrad = query.centroid
        gx, gy, grad = goal.centroid
        qxy = np.array((qx, qy))
        gxy = np.array((gx, gy))
        dist = np.linalg.norm(gxy - qxy)
        # Offset for intermediary points
        ratio = 0.5
        offset_length = ratio * dist
        # Two intermediary point A and B
        axy = qxy + offset_length * np.array((np.cos(qrad), np.sin(qrad)))
        # Heuristic score
        h_score = np.linalg.norm(gxy - axy) + np.linalg.norm(axy - qxy)
        return h_score

    @staticmethod
    def _se2_heuristic(query: PushHybridNode, goal: PushHybridNode) -> float:
        """Default heuristic function for Hybrid A*.

        This heuristic function assumes Z-shaped path to approximate a cubic Bézier curve.

        Args:
            query (PushHybridNode): Query node. It should be subclass of PushHybridNode.
            goal (PushHybridNode): Goal node. It should be subclass of PushHybridNode.

        Returns:
            float: Heuristic score from current node to goal
        """
        qx, qy, qrad = query.centroid
        gx, gy, grad = goal.centroid
        qxy = np.array((qx, qy))
        gxy = np.array((gx, gy))
        dist = np.linalg.norm(gxy - qxy)
        # Offset for intermediary points
        ratio = 0.5
        offset_length = ratio * dist
        # Two intermediary point A and B
        axy = qxy + offset_length * np.array((np.cos(qrad), np.sin(qrad)))
        bxy = gxy - offset_length * np.array((np.cos(grad), np.sin(grad)))
        # Heuristic score
        h_score = np.linalg.norm(gxy - bxy) + np.linalg.norm(bxy - axy) + np.linalg.norm(axy - qxy)
        return h_score

    @staticmethod
    def draw_goal(ax, shape, at, radius, style):
        if not isinstance(shape, (collision.Poly, collision.Concave_Poly)):
            raise TypeError("Shape must be a collision object of Poly or Concave_Poly")

        # get goal cetroid
        shape = copy.deepcopy(shape)
        shape.pos = collision.Vector(at[0], at[1])
        shape.angle = at[2]
        points = shape.points
        points = [[p.x, p.y] for p in points]
        points = np.array(points)
        goal_centroid = np.mean(points, axis=0)

        # draw goal centroid
        circle = plt.Circle((goal_centroid[0], goal_centroid[1]), radius, **style)
        ax.add_artist(circle)

    @staticmethod
    def draw_planning_scene(ax, start, goal, collision_system, search_space, drawing_bounds, draw_goal_shape):
        """Draw a planning scene
        
        Args:
            ax (matplotlib.axes.Axes): Axes to draw.
            start (tuple): The start position (x, y, theta) in world coordinate.
            goal (tuple): The goal position (x, y, theta) in world coordinate.
            collision_system (BoundingVolumeHierarchy): Supported collision system.
            search_space (DefaultHybridGrid): Supported search space.
            drawing_bounds (Tuple[float, float, float, float]): The (xmin, xmax, ymin, ymax) of the drawing area.
            draw_goal_shape (bool, optional): Animate the search. Defaults to False.
        """
        draw.draw_grid(ax, grid=search_space, drawing_bounds=drawing_bounds, style={"color": "0.8", "linewidth": 0.5})
        draw.draw_collsion_objects(ax, collision_system.obstacles, {"color": "0.5", "fill": True})

        # Draw background objects (agent-related objects-start)
        agent_shape = collision_system.agent_collision
        draw.draw_shape(ax, agent_shape, at=start, style={"color": pick_color(0.7, "turbo"), "fill": True})
        draw.draw_coordinates(ax, start, style={"color": "0", "head_width": 0.04, "head_length": 0.060, "coordinates_size": 0.12})
        
        # draw goal with shape
        if draw_goal_shape:
            draw.draw_shape(ax, agent_shape, at=goal, style={"color": pick_color(0.7, "turbo"), "fill": True})
            draw.draw_coordinates(ax, goal, style={"color": "0", "head_width": 0.04, "head_length": 0.060, "coordinates_size": 0.12})

        # draw goal
        HybridAstarPushPlanner.draw_goal(
            ax, agent_shape, at=goal, radius=0.05, style={"color": pick_color(0.7, "rainbow"), "fill": True})
        auto_scale(ax)
        
    @staticmethod
    def interpolate_path(path, number_of_waypoints):
        '''
        Interpolate the given path
        
        Inputs:
            path (2d numpy array): path to interpolate
            number_of_waypoints (int): number of waypoints in the path
            
    
        Inputs:
            interpolated_path (2d numpy array): interpolated path
        
        '''
        
        # Extract x, y, and theta columns from the motion path
        x = path[:, 0]
        y = path[:, 1]
        theta = path[:, 2]

        # Create an array of indices corresponding to the original motion path points
        N = len(path)
        indices = np.arange(N)

        # Create a new array of indices for the interpolated points
        new_indices = np.linspace(0, N-1, number_of_waypoints)

        # Perform cubic spline interpolation on x, y, and theta separately
        cs_x = CubicSpline(indices, x)
        cs_y = CubicSpline(indices, y)
        cs_theta = CubicSpline(indices, theta)

        # Evaluate the cubic splines at the new indices
        interpolated_x = cs_x(new_indices)
        interpolated_y = cs_y(new_indices)
        interpolated_theta = cs_theta(new_indices)

        # Combine the interpolated x, y, and theta into a new motion path
        interpolated_path = np.column_stack((interpolated_x, interpolated_y, interpolated_theta))
        
        return interpolated_path

    def _solve(self, start, goal, collision_system, search_space, animate=False):
        """Solve the Hybrid A* problem.

        Args:
            start (tuple): The start position (x, y, theta) in world coordinate.
            goal (tuple): The goal position (x, y, theta) in world coordinate.
            collision_system (BoundingVolumeHierarchy): Supported collision system.
            search_space (DefaultHybridGrid): Supported search space.
            animate (bool, optional): Animate the search. Defaults to False.

        Returns:
            path (list): The path with N x (x, y, theta) in world coordinate.
        """
        # get planner        
        planner = HybridAstar()
        planner.collision_system = collision_system
        planner.collision_system.build()
        planner.search_space = search_space
        planner.search_space.reset()
        
        if animate:
            # TODO: remove save option
            for i in range(0, 10):
                print('TODO: remove save option!!!')
            # save options
            fig_save_dir = '/home/cloudrobot2/Desktop/Figs'
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            before_plan_save_path = os.path.join(fig_save_dir, '{}_before.png'.format(time_str))
            after_plan_save_path = os.path.join(fig_save_dir, '{}_after.png'.format(time_str))
            
            ##########################################################
            # Draw planning scene before planning to check collision #
            ##########################################################
            # initialize
            print('Draw planning scene before the live demo.')
            fig, ax = plt.subplots()
            self.draw_planning_scene(
                ax, start, goal, planner.collision_system,
                planner.search_space, self._corners, draw_goal_shape=True)
            plt.show()

            ########################################################
            # Draw planning scene before planning to for the paper #
            ########################################################
            # TODO: remove it
            # save
            print('Save planning scene before the live demo.')
            fig, ax = plt.subplots()
            self.draw_planning_scene(
                ax, start, goal, planner.collision_system,
                planner.search_space, self._corners, draw_goal_shape=False)
            # plt.savefig(before_plan_save_path, dpi=1200)

            # set draw style
            """
            styles = {
                "focus_current_node": {
                    "color": "r",
                    "fill": False,
                    "coordinates_type": "directional circle",
                    "coordinates_size": 0.03,
                },
                "defocus_current_node": {
                    "color": "lightgray",
                    "fill": False,
                    "coordinates_type": "directional circle",
                    "coordinates_size": 0.03,
                },
                "open_list": {
                    "color": "gold",
                    "fill": False,
                    "coordinates_type": "directional circle",
                    "coordinates_size": 0.03,
                },
                "path_reconstruction": {
                    "color": "b",
                    "fill": True,
                    "coordinates_type": "directional circle",
                    "coordinates_size": 0.05,
                },
            }
            live_draw_options = {
                "focus_current_node": live.LiveDrawOption(
                    draw_func=lambda xyt: draw.draw_coordinates(ax, xyt, style=styles["focus_current_node"]),
                    # pause_after=0.05,
                    # wait_key=True,
                ),
                "defocus_current_node": live.LiveDrawOption(
                    draw_func=lambda xyt: draw.draw_coordinates(ax, xyt, style=styles["defocus_current_node"]),
                ),
                "open_list": live.LiveDrawOption(
                    draw_func=lambda xyt: draw.draw_coordinates(ax, xyt, style=styles["open_list"]),
                ),
                "path_reconstruction": live.LiveDrawOption(
                    draw_func=lambda xyt: draw.draw_coordinates(ax, xyt, style=styles["path_reconstruction"]),
                    pause_before=0.1,
                ),
            }

            # Compute path with live animation
            # --------------------------------
            # This is the only part that is different from the static version.

            planner.set_live_draw_options(live_draw_options)
            """

            waypoints = planner.solve(
                start, goal,
                fn_heuristic=self._cartesian_heuristic,
                fn_terminal_condition=self._cartesian_terminal_condition,)
            print(waypoints)
            print(len(waypoints))

            # Result
            # ------

            color = list(pick_color(0.3, "rainbow"))
            color[3] = 0.8  # Set alpha
            draw.draw_waypoints(
                ax,
                waypoints,
                shape=planner.collision_system.agent_collision,
                show_shape=True,
                shape_style={"color": color, "fill": False},
                show_coordinates=False,
            )
            # TODO: removeit
            # save
            # plt.savefig(after_plan_save_path, dpi=1200)

            # Wait for closing the plot
            plt.pause(0)
        else:
            waypoints = planner.solve(
                start, goal,
                fn_heuristic=self._cartesian_heuristic,
                fn_terminal_condition=self._cartesian_terminal_condition,)
        return waypoints

    def _plan(self, depth_image, contact_point, goal, learning_base=False, visualize=True):
        """Get stable push path from a given contact point and goal.

        Args:
            depth_image (numpy.ndarray): Depth image (H, W).
            contact_point (ContactPoint): ContactPoint object.
            goal (Tuple[float, float, float]): Goal X, Y, Theta.
            visualize (bool, optional): Whether to visualize the path. Defaults to False.

        Returns:
            path (List[Tuple[float, float, float]]): The path.
        """
        # Note: Planner plan with contact point frame 
        # but we want result with object frame
        # slider offset on the Node
        assert isinstance(contact_point, ContactPoint)
        slider_centroid_local = np.mean(contact_point.min_bbox, axis=0)
        PushHybridNode._centroid_offset = slider_centroid_local
        goal = (
            goal[0] - (np.cos(goal[2]) * slider_centroid_local[0] - np.sin(goal[2]) * slider_centroid_local[1]),
            goal[1] - (np.sin(goal[2]) * slider_centroid_local[0] + np.cos(goal[2]) * slider_centroid_local[1]),
            goal[2])

        # get collision system from contact point
        collision_system = self._get_collision_system(contact_point)
        print('start pose: {:.2f} {:.2f} {:.2f}'.format(
            contact_point.pose[0],
            contact_point.pose[1],
            np.rad2deg(contact_point.pose[2])))

        # get search space
        
        # TODO: self._dtheta is changed inside function. Fix it.
        if learning_base == True:
            left_min_radius, right_min_radius = self._get_stable_minimum_radius(
            depth_image, contact_point)
            
        else:
            left_min_radius, right_min_radius = self._get_stable_minimum_radius_depth_base(
                depth_image, contact_point)
        
        
        # --- TESTING: Getting maximum radius ---
        # left_max_radius, right_max_radius = self._get_stable_maximum_radius(
        #     depth_image, contact_point)
        # ---------------------------------------
        successor_template = self._get_successor_template(
            left_min_radius, right_min_radius)
        search_space = self._get_search_space(successor_template)

        # solve path
        print("Goal before solve: {:.2f} {:.2f} {:.2f}".format(
            goal[0], goal[1], goal[2]))
        path = self._solve(
            start=contact_point.pose, 
            goal=goal,
            collision_system=collision_system,
            search_space=search_space,
            animate=False)
        return path

    def update_map(self, map_corners, map_obstacles, **kwargs):
        """Update grid map.

        Args:
            map_corners (list): List of corners [xmin, xmax, ymin, ymax]
            map_obstacles (list): List of collision objects.
        """
        self._corners = map_corners
        self._obstacles = map_obstacles

        # if unknown arguments are given, raise an exception
        kwargs_keys = ['grid_size', 'dtheta']
        for key in kwargs.keys():
            if key not in kwargs_keys:
                raise ValueError(
                    'Unknown argument: {}'.format(key))

        # set grid size and direction resolution
        if 'grid_size' in kwargs.keys():
            self._grid_size = kwargs['grid_size']    
        if 'dtheta' in kwargs.keys():
            self._dtheta = kwargs['dtheta']

    def plan(self, depth_image, contact_points, goal, learning_base=False, visualize=False):
        """Plan a shortest stable push path from contact points.

        Args:
            depth_image (numpy.ndarray): Depth image in (H, W).
            contact_points (list of ContactPoint): List of ContactPoint objects.
            goal (list): Goal pose in world coordinate (x, y, theta).
            visualize (bool, optional): Whether to visualize the path. Defaults to False.

        Returns:
            path (numpy.ndarray): Stable push path in world coordinate (x, y, theta).
            contact_point (ContactPoint): ContactPoint object.
            success (bool): Whether the planning is successful.
        """
        contact_point_list = []
        path_list = []
        
        # Compare stability for all contact points
        # self._compare_stability_for_all_velocities(depth_image, contact_points)
        
        if visualize:
            contact_points[0].visualize_on_cartesian()
        
        for contact_point in contact_points:
            # print
            print('Plan with `({}, {}, {})` contact point'.format(
                contact_point.pose[0],
                contact_point.pose[1],
                np.rad2deg(contact_point.pose[2])))
 
            # get stable push path
            path = self._plan(
                depth_image, contact_point, goal,
                learning_base=learning_base,
                visualize=visualize)

            # append to list if successful
            if len(path) > 0:
                contact_point_list.append(contact_point)
                path_list.append(np.array(path))

        # get shortest path
        path_length_list = []
        for path in path_list:
            # path : (N, 3), (x, y, theta)
            path_length = 0
            for i in range(len(path) - 1):
                path_length += np.linalg.norm(path[i + 1, :2] - path[i, :2])
            path_length_list.append(path_length)
            
        try:
            shortest_path_idx = np.argmin(path_length_list)
        except:
            print('No successful path found')
            return [], False
        print('shortest_path_idx: ', shortest_path_idx)
        print('path_list len: ', len(path_list))

        shortest_path = path_list[shortest_path_idx]
        
        # Interpolate the path with 100 waypoints. 
        # This is because Doosan Robot Arm accepts a path with maximum 100 waypoints.
        interpolated_path = self.interpolate_path(shortest_path, number_of_waypoints=100)
        
        return interpolated_path, True
class RRTPushPlanner(object):
    def __init__(self):
        pass

    def plan(self, start, goal):
        pass
class MinimumICRPushPlanner(object):
    def __init__(self, direction):
        """Initialize minimum icr push planner.

        Args:
            direction (float): contact direction 0-2*pi. unit: radian.
        """
        self.direction = direction

    @staticmethod
    def _get_min_icr_push_path(visualization=False):
        """Get minimum icr push path.

        Args:
            visualization (bool): if True, visualize the path.

        Returns:
            path (numpy.ndarray): path (N, 3) with (x, y, theta). unit: meter, meter, radian.
        """
        large_radius = [100, 90, 80, 70, 60, 50, 40, 30, 20,
                        10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        small_radius = np.linspace(1, 0.001, 70)
        radius_list = np.concatenate((large_radius, small_radius))
        ds = 0.005  # unit: m

        x_list = [0.0]
        y_list = [0.0]
        th_list = [0.0]
        for i, radius in enumerate(radius_list):
            # stear angle
            dtheta = ds / radius
            # if stear angle is too large, limit it
            if dtheta > np.pi/9:
                dtheta = np.pi/9

            # get dx, dy in local frame
            dx = radius * np.sin(dtheta)
            dy = radius * (1 - np.cos(dtheta))

            # transform dx, dy to global frame
            rot_mat = np.array([
                [np.cos(th_list[i]), -np.sin(th_list[i])],
                [np.sin(th_list[i]), np.cos(th_list[i])]])
            dx, dy = np.matmul(rot_mat, np.array([dx, dy]))

            # update x, y, theta
            x_list.append(x_list[i] + dx)
            y_list.append(y_list[i] + dy)
            th_list.append(th_list[i] + dtheta)

        # plot car moving as arrow, y-axis is car direction
        if visualization:
            th_list = np.array(th_list)
            th_list = th_list + np.pi/2
            plt.figure()
            plt.plot(x_list, y_list)
            for i in range(len(x_list)):
                plt.arrow(x_list[i], y_list[i], 0.005*np.cos(th_list[i]), 0.005*np.sin(th_list[i]), color='r')
            plt.axis('equal')
            plt.show()

        path = np.array([x_list, y_list, th_list]).T
        return path

    @staticmethod
    def _xyt_to_se2(xyt):
        """Convert (x, y, theta) to SE2.

        Args:
            xyt (numpy.ndarray): (x, y, theta). unit: meter, meter, radian.

        Returns:
            se2 (numpy.ndarray): SE2 matrix.
        """
        # xyt: (x, y, theta)
        se2 = np.array([
            [np.cos(xyt[2]), -np.sin(xyt[2]), xyt[0]],
            [np.sin(xyt[2]), np.cos(xyt[2]), xyt[1]],
            [0, 0, 1]])
        return se2

    @staticmethod
    def _transform_path(init_pose, path):
        """Transform path from init pose to global frame.

        Args:
            init_pose (numpy.ndarray): (x, y, theta). unit: meter, meter, radian.
            path (numpy.ndarray): path (N, 3) with (x, y, theta). unit: meter, meter, radian.

        Returns:
            transformed_path (numpy.ndarray): path (N, 3) with (x, y, theta). unit: meter, meter, radian.
        """        
        # init pose: (x, y, theta)
        # path: [(x, y, theta), ...]
        # return: [(x, y, theta), ...]
        init_pose = MinimumICRPushPlanner._xyt_to_se2(init_pose)
        transformed_path = []
        for xyt in path:
            se2 = MinimumICRPushPlanner._xyt_to_se2(xyt)
            transformed_se2 = np.matmul(init_pose, se2)
            transformed_xyt = np.array([
                transformed_se2[0, 2],
                transformed_se2[1, 2],
                np.arctan2(transformed_se2[1, 0], transformed_se2[0, 0])])
            transformed_path.append(transformed_xyt)
        transformed_path = np.array(transformed_path)
        return transformed_path

    def plan(self, depth_image, contact_points, goal, visualization=False):
        """Plan path.

        Args:
            depth_image (numpy.ndarray): depth image (H, W).
            contact_points (numpy.ndarray): contact points (N, 3).
            goal (numpy.ndarray): goal (N, 3).

        Returns:
            path (numpy.ndarray): (N, 3) with (x, y, theta). unit: meter, meter, radian.
        """
        push_dirs = np.array([cp.push_direction for cp in contact_points])
        angles = np.arctan2(push_dirs[:,1], push_dirs[:,0])
        push_arg = np.argmin(np.abs(self.direction - angles))
        contact_point = contact_points[push_arg]
        
        assert isinstance(contact_point, ContactPoint)
        
        init_pose = contact_point.pose
        
        # get path
        init_path = self._get_min_icr_push_path(visualization=False)
        transformed_path = self._transform_path(init_pose, init_path)

        # visualize on depth image
        # contact_point1 color: red
        # contact_point2 color: green
        visualization = True
        if visualization:
            contact_point.visualize_contact_point(depth_image)

        # plot car moving as arrow
        if visualization:
            x_list = transformed_path[:, 0]
            y_list = transformed_path[:, 1]
            th_list = transformed_path[:, 2]
            th_list = th_list

            plt.figure()
            plt.plot(
                contact_point.edge_xyz[:, 0],
                contact_point.edge_xyz[:, 1], 'kx')
            plt.plot(x_list, y_list, 'b-')
            for i in range(len(x_list)):
                plt.arrow(
                    x_list[i], y_list[i],
                    0.005 * np.cos(th_list[i]),
                    0.005 * np.sin(th_list[i]),
                    color='r')
            plt.axis('equal')
            plt.show()

        return transformed_path
 
    def update_map(self, map_corners, map_obstacles, **kwargs):
        pass

# if __name__ == '__main__':
    
#     with open("/home/cloudrobot2/catkin_ws/src/push_planners/twc-stable-pushnet/config/datagen.yaml", "r") as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)
#     cfg = cfg['simulation']
#     objects = ["melamineware_g_0151", "melamineware_g_0130", "dish_small", "rice_bowl", "soup_bowl"]
#     num_envs, num_iters = cfg["num_envs"], cfg["num_iters"]
#     cam_cfg = cfg["camera"]["ZividTwo"]
#     fx, fy, cx, cy = cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"]
#     camera_intr = np.array([[fx,0,cx], [0,fy,cy], [0, 0, 1]])
    
#     stable_determinator = StablePushNetDeterminator()
#     # stable_determinator = DepthImageBasedDeterminator()
#     planner = HybridAstarPushPlanner(stable_determinator)
#     # For each object
#     for object_name in objects:
#         # For each depth image
#         for file_idx in range(num_envs * num_iters):
#             cam_pose_name = ("cam_pose_" + object_name + "_%05d.npy")%(file_idx)
#             mask_name     = ("mask_" + object_name + "_%05d.npy")%(file_idx)
#             depth_name    = ("image_" + object_name + "_%05d.npy")%(file_idx)
#             camera_extr   = np.load("../data/input_images/" + cam_pose_name, allow_pickle=True)
#             segmentation_mask = np.load("../data/input_images/" + mask_name, allow_pickle=True)
#             depth_image   = np.load("../data/input_images/" + depth_name, allow_pickle=True)
#             # Sample contact points
#             contact_point_sampler = ContactPointSampler(camera_intr, camera_extr)
#             contact_points = contact_point_sampler.sample(depth_image, segmentation_mask)
            
#             goal_center = np.array([30, 30])
#             obstacles_and_map = np.arange(100).reshape(-1,2)
#             # Sample contact points
#             best_contact_point, best_trajectory = planner.plan(depth_image, contact_points, goal_center)
