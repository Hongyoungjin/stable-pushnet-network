import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .stable_region_analytical import StableRegion
from scipy import ndimage
import cv2
import yaml

def fibonacci_sphere(samples=2000):

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples//2):
        x = 1 - (i / float(samples - 1)) * 2  # x goes from 1 to -1
        radius = np.sqrt(1 - x * x)  # radius at x

        theta = phi * i  # golden angle increment

        y = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))
    velocity = np.array(points)
    return velocity
    
def velocity2icr(velocity):
    """
    Calculate ICR (Instantaneous Center of Rotation) for each velocity.
    """
    vx, vy, w = velocity[:,0], velocity[:,1], velocity[:,2]
    ICRs = []
    for i in range(len(vx)):
        if w[i] == 0:
            # icr= np.array([np.inf, np.inf])
            w[i] = 1e-6
        # else:
        icr= np.array([-vy[i] / w[i], vx[i] / w[i]])
        ICRs.append(icr)
        
    return np.array(ICRs)
    
def crop_image(depth_image, push_contact):
        ''' Convert the given depth image to the network image input
        
        1. Set the contact point to the center of the image
        2. Rotate the image so that the push direction is aligned with the x-axis
        3. Crop the image so that the object can be fit into the entire image
        4. Resize the image to the network input size (96 x 96)
        
        '''
        
        image_height, image_width = 96, 96

        H,W = depth_image.shape
        contact_points_uv = push_contact.contact_points_uv
        edge_uv = push_contact.edge_uv
        edge_center_uv = edge_uv.mean(axis=0)
        
        '''
        contact_points_uv, edge_uv: [row, col] = [u,v]
        Image coordinate:           [row, col] = [v,u]
        '''
        
        contact_center_uv = contact_points_uv.mean(0).astype(int)
        contact_center_vu = np.array([contact_center_uv[1], contact_center_uv[0]])
        
        ########################################################
        # Modify pushing direction to head to the -v direction #
        ########################################################
        u1, v1 = contact_points_uv[0]
        u2, v2 = contact_points_uv[1]
        push_dir = np.array([u1-u2,v2-v1])
        
        # Center of the rotated edge center should be in -v direction
        rot_rad = np.pi - np.arctan2(push_dir[1],push_dir[0])  # push direction should head to the -v direction (up in the image)
        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], 
                      [np.sin(rot_rad),  np.cos(rot_rad)]])
        edge_center_vu = np.array([edge_center_uv[1], edge_center_uv[0]])
        rotated_edge_center = R @ (edge_center_vu - contact_center_vu)
        
        if rotated_edge_center[0] > 0:
            rot_angle = 180 + np.rad2deg(rot_rad)
        else:
            rot_angle = np.rad2deg(rot_rad)
            
        ###################################
        # Rotate and crop the depth image #
        ###################################
        
        # Shift the image so that the contact point is at the center
        shifted_img = ndimage.shift(depth_image, (np.round(H/2-contact_center_vu[0]).astype(int), np.round(W/2-contact_center_vu[1]).astype(int)), mode='nearest')
        
        # Rotate the image so that the pushing direction heads to -v direction
        rotated_img = ndimage.rotate(shifted_img, rot_angle, mode='nearest', reshape=False)
        
        
        # Crop the image so that the object can be fit into the entire image
        center_y, center_x = np.round(H/2).astype(int), np.round(W/2).astype(int)
        crop_size_unit = 75
        cropped_img = rotated_img[center_y - 3*crop_size_unit : center_y + crop_size_unit, center_x  - 2*crop_size_unit : center_x  + 2*crop_size_unit]
        
        # Resize the image to the network input size (96 x 96)
        cropped_img = cv2.resize(cropped_img, (image_width,image_height))
        
        return cropped_img
    
    
    