import numpy as np
import collision
from corgipath.collision import BoundingVolumeHierarchy
from corgipath.search_space import DefaultHybridGrid, DefaultHybridNode, HybridSuccessor
from corgipath.planning import HybridAstar
from corgipath.matplot import static_draw as draw
from corgipath.matplot import live_draw as live
from corgipath.matplot.utils import pick_color, auto_scale


class HybridAstarTrajectory(object):
    def __init__(self, grid_size, direction_resolution):
        self.planner = HybridAstar()
        self.grid_size = grid_size
        self.direction_resolution = direction_resolution
        
    def get_inequal_successor_template(self, max_heading_change_left: float, max_heading_change_right: float):
        """Set the successor template to the default template.

        Args:
            max_heading_change_left (float): The maximum heading change allowed in the template. The unit is radian.
            max_heading_change_right (float): The maximum heading change allowed in the template. The unit is radian.
        """
        assert max_heading_change_left >= 0.0
        assert max_heading_change_right >= 0.0

        # In Hybrid A*, minimum distance to forward >= diagonal length.
        diagonal = np.sqrt(2.0) * self.grid_size
        # Pair of displacement and cost
        forward = (diagonal, diagonal)
        edges = (forward,)
        # Make template
        template = []
        for displacement, cost in edges:
            template.append(HybridSuccessor.from_heading_with_dist(0.0, displacement, cost))
            _dist = displacement / np.sqrt(2.0)
            _cost = cost / np.sqrt(2.0)
            for i in range(1, int(max_heading_change_left / self.direction_resolution) + 1):
                template.append(HybridSuccessor.from_heading_with_dist(self.direction_resolution * i, _dist, _cost))
            for i in range(1, int(max_heading_change_right / self.direction_resolution) + 1):
                template.append(HybridSuccessor.from_heading_with_dist(-self.direction_resolution * i, _dist, _cost))
        return template
    
    def get_collision_system(self, slider_collision, corners, obstacles):
        """get the collision system.

        Args:
            slider_collision (numpy.ndarray): (N, 2) polygon points of slider collision.
            corners (Tuple[float, float, float, float]): The bounds of the environment.
                The format is (x_min, x_max, y_min, y_max).
            obstacles (List[collision.*]): List of collision objects.

        Returns:
            bounding_volume_hierarchy (BoundingVolumeHierarchy): Bounding volume hierarchy.
        """
        # Define static objects (obstacles) here.
        bvh = BoundingVolumeHierarchy(bounds=corners)
        bvh.add_obstacles(obstacles)
        bvh.build()

        # X-axis is forward in the agent's frame
        bvh.agent_collision = collision.Poly(
            collision.Vector(0, 0),
            [collision.Vector(p[0], p[1]) for p in slider_collision])
        return bvh

    def get_search_space(self, grid_size, theta_gap, icr_left, icr_right):
        """Define the search space.

        Args:
            grid_size (float): The grid size of the search space.
            theta_gap (float): The gap between two adjacent angles.
            icr_left (numpy.ndarray): [icr_x, icr_y] for left side.
            icr_right (numpy.ndarray): [icr_x, icr_y] for right side.
            
        Returns:
            search_space (DefaultHybridGrid): The search space.
        """
        # get radius from icr
        radius_left = abs(icr_left[1])
        radius_right = abs(icr_right[1])

        # get max heading change
        max_heading_change_left = np.sqrt(2.0) / radius_left
        max_heading_change_right = np.sqrt(2.0) / radius_right
        successor_template = self.get_inequal_successor_template(
            max_heading_change_left=max_heading_change_left,
            max_heading_change_right=max_heading_change_right)
        print("Set successor template:\n\tmax_heading_change_left: {}\n\tmax_heading_change_right: {}".format(
            np.rad2deg(max_heading_change_left),
            np.rad2deg(max_heading_change_right)))

        # Define the search space
        grid = DefaultHybridGrid(dxy=grid_size, dtheta=theta_gap, node_type=DefaultHybridNode)
        grid.successor_template = successor_template
        return grid

    def plan(self, start, goal, icr_left, icr_right, slider_collision, corners, obstacles, visualization=False):

        """Plan a trajectory.

        Args:
            start (Tuple[float, float, float]): Start X, Y, Theta
            goal (Tuple[float, float, float]): Goal X, Y, Theta
            icr_left (_type_): _description_
            icr_right (_type_): _description_
            slider_collision (_type_): _description_
            corners (_type_): _description_
            obstacles (_type_): _description_
            visualization (bool, optional): _description_. Defaults to False.

        Returns:
            List[Tuple[float, float, float]]: Waypoints (List of X, Y, Theta). Empty list if no solution is found.
        """
        bvh = self.define_collision_system(slider_collision, corners, obstacles)
        self.define_search_space(bvh, icr_left, icr_right)
        self.planner.search_space.reset()
        
        
        
        waypoints = self.planner.solve(start, goal)
        
        return waypoints