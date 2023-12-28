#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import ConvexHull
from .tools import polygon_centroid, Rmat2D, Tmat2D, TmatDot, TmatDotBulk

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
def unit(rad):
    return np.array([np.cos(rad), np.sin(rad)])


class StableRegion:
    HALF_PI = np.pi / 2.0

    def __init__(self, input_points, contact_points, contact_normals, default_mu=0.5):
        """
        2D stable push region determinator.

        Args:
            input_points (numpy.ndarray): (N, 2) edge points with (x, y) world coordinates.
            contact_points (numpy.ndarray): (2, 2) contact points. Each row is a contact point (x, y) world coordinates.
            contact_normals (numpy.ndarray): (2, 2) contact normals. Each row is a contact normal (x, y) world coordinates.
            mu (float): friction coefficient.
        """
        # preprocess input points(convex hull)
        cvh = ConvexHull(input_points)
        input_points = input_points[cvh.vertices, :]

        # Input coordinates
        self._xy_points = None
        self._centroid = None  # _local_centroid of friction
        self._line_contact_cases = None
        self._current_contact = None
        self._local_centroid_input = np.mean(input_points, axis=0)
        # Transformation
        self._local_input_T = None
        self._input_local_T = None
        self._input_local_R = None

        self._mu = default_mu

        # Pusher coordinates (centor of pusher)
        self._local_xy_points = None
        self._local_centroid = None
        self._local_lsupport = None  # xy
        self._local_rsupport = None  # xy

        # Stable conditions
        """
        Reference: Page 159 of "Mechanics of robotic manipulation"
        (M. T. Mason, Mechanics of robotic manipulation. Cambridge, Mass: MIT Press, 2001.)
        """
        """
        # 1: Friction cone condition (mu & shape dependent)
        -------------------------------------------------
        Left-hand ICR is stable if:
            y>= left cone의 -alpha (x=상수꼴이면 queryX<=x)
            y>= right cone의 alpha + 떨어진점 (x=상수꼴이면 queryX>=x)
        Right-hand ICR is stable if:
            y<= left cone의 -alpha + 떨어진점 (x=상수꼴이면 queryX>=x)
            y<= right cone의 alpha (x=상수꼴이면 queryX<=x)
        """
        self._cond_FL11 = LineConstraint(greater_than_y=True, greater_than_x=False)
        self._cond_FL12 = LineConstraint(greater_than_y=True, greater_than_x=True)
        self._cond_FR11 = LineConstraint(greater_than_y=False, greater_than_x=True)
        self._cond_FR12 = LineConstraint(greater_than_y=False, greater_than_x=False)
        
        self._cond_FL21 = LineConstraint(greater_than_y=True, greater_than_x=False)
        self._cond_FL22 = LineConstraint(greater_than_y=True, greater_than_x=True)
        self._cond_FR21 = LineConstraint(greater_than_y=False, greater_than_x=True)
        self._cond_FR22 = LineConstraint(greater_than_y=False, greater_than_x=False)

        """
        # 2: Wrench condition (= Non-prehensile condition)) (shape dependent)
        -------------------------------------------------
        (r: centroid에서 가장 먼 cone까지의 거리, p: centroid에서 현재 cone까지의 거리)
        Left-hand ICR is stable if:
            y>= right cone과 centroid 사이의 수직이등분선 (x=상수꼴이면 queryX<=x)
            y>= `(현재)left cone->centroid`방향, centroid에서 r^2/p 거리 (x=상수꼴이면 queryX>=x)
        Right-hand ICR is stable if:
            y<= `(현재)right cone->centroid`방향, centroid에서 r^2/p 거리 (x=상수꼴이면 queryX>=x)
            y<= left cone과 centroid 사이의 수직이등분선 (x=상수꼴이면 queryX<=x)
        """
        self._cond_WL1 = LineConstraint(greater_than_y=True, greater_than_x=False)
        self._cond_WL2 = LineConstraint(greater_than_y=True, greater_than_x=True)
        self._cond_WR1 = LineConstraint(greater_than_y=False, greater_than_x=True)
        self._cond_WR2 = LineConstraint(greater_than_y=False, greater_than_x=False)
        
        self.init_slider(input_points, contact_points, contact_normals)

    @property
    def mu(self):
        return self._mu

    @property
    def local_input_Tmat(self):
        return self._local_input_T

    @property
    def local_input_Rmat(self):
        return self._input_local_R.transpose()

    @property
    def local_forward_vector(self):
        return np.dot(self.local_input_Rmat, np.array((1, 0)))

    @property
    def local_xy_points(self):
        return self._local_xy_points

    @property
    def local_centroid(self):
        return self._local_centroid

    @property
    def local_lsupport(self):
        return self._local_lsupport

    @property
    def local_rsupport(self):
        return self._local_rsupport

    @property
    def stable_constraints_of_friction(self):
        return (self._cond_FL1, self._cond_FL2, self._cond_FR1, self._cond_FR2)

    @property
    def stable_constraints_of_wrench(self):
        return (self._cond_WL1, self._cond_WL2, self._cond_WR1, self._cond_WR2)

    def init_slider(self, input_points, contact_points, contact_normals):
        """
        input_points: list of xy points of object contour. [(x,y), (x,y), ...]
        contact_points: list of contact points. [L(x,y), R(x,y)]
        """
        if not isinstance(input_points, np.ndarray):
            input_points = np.array(input_points)
        self._input_points = input_points
        if not isinstance(contact_points, np.ndarray):
            contact_points = np.array(contact_points)
        self.contact_points = contact_points
        self.contact_normals = contact_normals
        # Friction _local_centroid (cof) == _local_centroid of gravity (centroid)
        self._centroid = polygon_centroid(input_points)
        self.set_current_contact()

    def convert_coordinate(self):
        input_localOrigin = (self.input_lsupport + self.input_rsupport) / 2.0
        input_localYvec = self.input_lsupport - self.input_rsupport
        input_localXvec = TmatDot(Tmat2D(-self.HALF_PI, 0, 0), input_localYvec)
        rad = np.arctan2(input_localXvec[1], input_localXvec[0])

        # Set pusher coordinates
        self._input_local_T = Tmat2D(rad, *input_localOrigin)
        self._local_input_T = np.linalg.inv(self._input_local_T)
        self._input_local_R = Rmat2D(rad)

        # Pusher coordinates
        self._local_xy_points = TmatDotBulk(self._local_input_T, self._input_points)
        self._local_centroid = TmatDot(self._local_input_T, self._centroid)
        self._local_lsupport = TmatDot(self._local_input_T, self.input_lsupport)
        self._local_rsupport = TmatDot(self._local_input_T, self.input_rsupport)
        self._local_lnormal  = Rmat2D(-rad) @ self.input_lnormal
        self._local_rnormal  = Rmat2D(-rad) @ self.input_rnormal
        
    def set_current_contact(self):
        # Support points in the input frame
        self.input_lsupport = self.contact_points[0]
        self.input_rsupport = self.contact_points[1]
    
        # Normals in the input frame
        self.input_lnormal = self.contact_normals[0]
        self.input_rnormal = self.contact_normals[1]
        
        self.convert_coordinate()
        if self._local_centroid[0] < 0.0:
            # Support points in the input frame
            self.input_lsupport = self.contact_points[1]
            self.input_rsupport = self.contact_points[0]
        
            # Normals in the input frame
            self.input_lnormal = self.contact_normals[1]
            self.input_rnormal = self.contact_normals[0]
            self.convert_coordinate()
        
        # self.show_points()
        # UPDATE CONSTRAINTS ##########################
        # 1: Friction cone condition (mu & shape dependent)
        self.update_friction_cone_stable()
        # # 2: Wrench condition (= Non-prehensile condition)) (shape dependent)
        self.update_wrench_stable()
        
        return self._current_contact


    def plot_constraints(self):
        constraints = [self._cond_FL11 ,self._cond_FL12 ,self._cond_FR11 ,self._cond_FR12,
         self._cond_FL21 ,self._cond_FL22 ,self._cond_FR21 ,self._cond_FR22]
        
        starting_points = np.array([constraint.point for constraint in constraints])
        heading_directions = [[1, constraint._m] for constraint in constraints]
        heading_directions = np.array(heading_directions)
        unit_heading_directions = heading_directions / np.linalg.norm(heading_directions)
        end_points = starting_points + unit_heading_directions *10
        starting_points = starting_points - unit_heading_directions *10
        
        lines = []
        for i in range(len(starting_points)):
            starting_point, end_point = starting_points[i], end_points[i]
            line = [(starting_point[0], starting_point[1]), (end_point[0], end_point[1])]
            lines.append(line)
        lines = LineCollection(lines)
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.5,0.5)
        ax.add_collection(lines)
        ax.scatter(self._local_xy_points[:,0], self._local_xy_points[:,1], s=0.5)
        ax.set_aspect('equal')
        plt.show()
    
            
            
    def update_friction(self, mu):
        """
        mu: friction coefficient (mu = tan(alpha))
        """
        self._mu = mu
        # UPDATE CONSTRAINTS ##########################
        # 1: Friction cone condition (mu & shape dependent)
        self.update_friction_cone_stable()

    def friction_condition(self, local_xy, constraints):
        angles = np.array([constraints[0].slope_angle, constraints[1].slope_angle, constraints[2].slope_angle, constraints[3].slope_angle])
        min, max = constraints[np.argmin(angles)], constraints[np.argmax(angles)]
        min_angle, max_angle = min.slope_angle, max.slope_angle
        self.origin = self._intersection(self.vector2point(min.point, min.slope_angle), self.vector2point(max.point, max.slope_angle))
        eval_angle = np.arctan2((local_xy - self.origin)[1], (local_xy - self.origin)[0])
        
        # Exception for the left-hand side constraint
        if max_angle >= np.pi:
            max_angle -= 2*np.pi
            condition = any((min_angle <= eval_angle, eval_angle <= max_angle))
        # Exception for the right-hand side constraint
        elif min_angle <= -np.pi:
            min_angle += 2*np.pi
            condition = any((min_angle <= eval_angle, eval_angle <= max_angle))
        # Default case
        else:
            condition = all((min_angle <= eval_angle, eval_angle <= max_angle))
        
        return condition
    
    def is_stable_in_local_frame(self, local_xy):
        
        # Left-hand ICR
        constraints_L = [self._cond_FR11, self._cond_FR12, self._cond_FR21, self._cond_FR22]
        condition_L = self.friction_condition(local_xy, constraints_L)
        # Right-hand ICR
        constraints_R = [self._cond_FL11, self._cond_FL12, self._cond_FL21, self._cond_FL22]
        condition_R = self.friction_condition(local_xy, constraints_R)
        
        # Wrench constraints
        if local_xy[1] < 0.0:  # Right-hand ICR
            conditions_W = (self._cond_WR1.is_stable(local_xy),self._cond_WR2.is_stable(local_xy))
            
        else: # Left-hand ICR
            conditions_W = (self._cond_WL1.is_stable(local_xy),self._cond_WL2.is_stable(local_xy))
        
        is_stable = all((any((condition_L,condition_R)),all(conditions_W)))
        
        return is_stable
    
    def update_friction_cone_stable(self):
        # Friction cone slope: m = tan(alpha) = friction coefficient mu
        self.alpha = np.arctan(self._mu)
        self.lsup = self._local_lsupport
        self.rsup = self._local_rsupport
        self._lrad = np.arctan2(self._local_lnormal[1], self._local_lnormal[0])
        self._rrad = np.arctan2(self._local_rnormal[1], self._local_rnormal[0])
        
        #######################
        ## Left contact point##
        #######################
        # Farthest points
        self.FL11_point = self._farthest_local_xy(self._local_centroid,  unit(self._lrad - self.alpha))
        self.FL12_point = self._farthest_local_xy(self._local_centroid, -unit(self._lrad + self.alpha))
        self.FR11_point = self._farthest_local_xy(self._local_centroid, -unit(self._lrad - self.alpha))
        self.FR12_point = self._farthest_local_xy(self._local_centroid,  unit(self._lrad + self.alpha))
        
        # Angles for constraint 
        self.FL11_angle = self._lrad - self.alpha + self.HALF_PI
        self.FL12_angle = self._lrad + self.alpha + self.HALF_PI
        self.FR11_angle = self._lrad - self.alpha - self.HALF_PI
        self.FR12_angle = self._lrad + self.alpha - self.HALF_PI
        
        # Constraints for the friction cone 
        self._cond_FL11.update(self.FL11_point, self.FL11_angle)
        self._cond_FL12.update(self.FL12_point, self.FL12_angle)
        self._cond_FR11.update(self.FR11_point, self.FR11_angle)
        self._cond_FR12.update(self.FR12_point, self.FR12_angle)
        
        ########################
        ## Right contact point##
        ########################
        
        # Farthest points for the right contact point
        self.FL21_point = self._farthest_local_xy(self._local_centroid,  unit(self._rrad - self.alpha))
        self.FL22_point = self._farthest_local_xy(self._local_centroid, -unit(self._rrad + self.alpha))
        self.FR21_point = self._farthest_local_xy(self._local_centroid, -unit(self._rrad - self.alpha))
        self.FR22_point = self._farthest_local_xy(self._local_centroid,  unit(self._rrad + self.alpha))
        
        # Angles for constraint
        self.FL21_angle = self._rrad - self.alpha + self.HALF_PI
        self.FL22_angle = self._rrad + self.alpha + self.HALF_PI
        self.FR21_angle = self._rrad - self.alpha - self.HALF_PI
        self.FR22_angle = self._rrad + self.alpha - self.HALF_PI
        
        # Constraints for the friction cone 
        self._cond_FL21.update(self.FL21_point, self.FL21_angle)
        self._cond_FL22.update(self.FL22_point, self.FL22_angle)
        self._cond_FR21.update(self.FR21_point, self.FR21_angle)
        self._cond_FR22.update(self.FR22_point, self.FR22_angle)
        
        
    def vector2point(self, point, direction):
        vector = np.array([np.cos(direction), np.sin(direction)])
        return [point, point + vector]
    
    def _intersection(self, line1, line2):
        # line intersection btw line defined by (x1,y1) and (x2,y2) and line defined by (x3,y3) and (x4,y4)
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
            point1, point2, point3, point4 = line1[0],line1[1], line2[0],line2[1]
            x1, y1, x2, y2, x3, y3, x4, y4 = point1[0], point1[1], point2[0], point2[1], point3[0], point3[1], point4[0], point4[1]
            px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
            py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
            
            return np.array([px, py])
    def _farthest_local_xy(self, from_xy, direction):
        """
        from_xy: xy point in local frame
        direction: We need a farthest point in this direction.
        ---
        Find a point that maximizes the dot product with the direction vector..
        """
        self.ignore=0
        farthest_point = None
        farthest_dot = 0
        for xy in self._local_xy_points:
            dot = np.dot(xy - from_xy, direction)
            if dot > farthest_dot:
                farthest_point = xy
                farthest_dot = dot

        return farthest_point

    def update_wrench_stable(self):
        # Passpoint and angle_rad of Perpendicular bisectors
        # Wrench Left 1 (left-cone, centroid)
        lper = self._perpendicular_bisector(self._local_lsupport, self._local_centroid)
        self._cond_WL1.update(*lper)
        # Wrench Right 2 (right-cone, centroid)
        rper = self._perpendicular_bisector(self._local_rsupport, self._local_centroid)
        self._cond_WR2.update(*rper)

        # Passpoint and angle_rad of Farpoint Lines
        _ldiff = self._local_centroid - self._local_lsupport  # != 0
        _rdiff = self._local_centroid - self._local_rsupport  # != 0
        _ldiff_sqdist = np.dot(_ldiff, _ldiff)
        _rdiff_sqdist = np.dot(_rdiff, _rdiff)
        _r_square = max(_ldiff_sqdist, _rdiff_sqdist)
        # Wrench Right 1 (left-cone -> centroid -> line)
        """
        _ldiff_length = np.sqrt(_ldiff_sqdist)
        _dist = _r_square / _ldiff_length  # r^2 / p
        dxy = unit_vector * _dist  # dxy_from_centroid
            = (_ldiff / _ldiff_length) * _dist
        -----------------------------
        This is the same as below:
        dxy = (_ldiff / _ldiff_length) * (_r_square / _ldiff_length)
        dxy = _ldiff * (_r_square / _ldiff_sqdist)
        """
        dxy = (_r_square / _ldiff_sqdist) * _ldiff
        farpoint1 = self._local_centroid + dxy
        self._cond_WR1.update(farpoint1, lper[1])  # passpoint, angle
        # Wrench Left 2 (right -> centroid -> line)
        dxy = (_r_square / _rdiff_sqdist) * _rdiff
        farpoint2 = self._local_centroid + dxy
        self._cond_WL2.update(farpoint2, rper[1])  # passpoint, angle

    def _perpendicular_bisector(self, point1, point2):
        """
        point1, point2: 2 points in local frame
        ---
        Return the perpendicular bisector of the line segment between point1 and point2.
        => (pass_point, angle_rad)
        """
        direction = point2 - point1
        # +90 degree is
        # rotated = np.array([-direction[1], direction[0]])
        # slope = -direction[0] / direction[1]
        angle_rad = (
            np.arctan(-direction[0] / direction[1])
            if direction[1] != 0
            else self.HALF_PI
        )
        pass_point = (point1 + point2) / 2.0
        return (pass_point, angle_rad)


class LineConstraint:
    MAX_RAD = np.pi / 2.0

    def __init__(self, greater_than_y=True, greater_than_x=True):
        self._greater_than_y = greater_than_y
        self._greater_than_x = greater_than_x
        self._m = None
        self._b = None
        self._x1 = None
        self.slope_angle = 0
        self.point = np.zeros(2)
        self._is_caseA = None

    @property
    def is_caseA(self):
        return self._is_caseA

    @property
    def is_caseB(self):
        return not self._is_caseA

    @property
    def x1(self):
        return self._x1

    @property
    def standard_form(self):
        """ax + by + c = 0"""
        # case A: mx - y + b = 0
        # case B: x      - x1 = 0
        if self._is_caseA:
            return (self._m, -1.0, self._b)
        else:
            return (1.0, 0.0, -self._x1)

    def update(self, pass_point, angle_rad):
        """
        pass_point: (x1,y1)
        slope: m = tan(angle)
        Point-Slope form: y-y1 = m(x-x1)
        Slope-Intercept form: y = mx + b
        """

        self.slope_angle = angle_rad
        self.point = pass_point
        self._m = np.tan(angle_rad)
        self._b = pass_point[1] - self._m * pass_point[0]
        self._x1 = pass_point[0]

        # -np.pi/2 <= angle_rad <= np.pi/2
        # case A: abs(angle_rad) < np.pi/2  => Check Y
        # case B: abs(angle_rad) == np.pi/2 => Check X
        abs_rad = abs(angle_rad)
        self._is_caseA = abs_rad < self.MAX_RAD
        if abs_rad > self.MAX_RAD:
            Exception("Angle must be in range of [-pi/2, pi/2]")
            
    def is_left(self, xy):
        return self.is_stable(xy)

    def is_stable(self, xy):
        if self._is_caseA:
            return self._is_stable_caseA(xy)
        else:
            return self._is_stable_caseB(xy)

    def _is_stable_caseA(self, xy):
        if self._greater_than_y:
            return xy[1] >= self._m * xy[0] + self._b
        else:
            return xy[1] <= self._m * xy[0] + self._b

    def _is_stable_caseB(self, xy):
        if self._greater_than_x:
            return xy[0] >= self._x1
        else:
            return xy[0] <= self._x1
