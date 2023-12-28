#!/usr/bin/env python
# -*- coding: utf-8 -*-
import yaml
import torch
import numpy as np
from .utils.model import PushNet
from .utils.utils import fibonacci_sphere, crop_image, velocity2icr
from .utils.stable_region_analytical import StableRegion
import matplotlib.pyplot as plt
import os
import rospy
torch.multiprocessing.set_sharing_strategy('file_system')


class StablePushNetDeterminator(object):
    def __init__(self):
        
        # Get current file path
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.abspath(os.path.join(current_file_path, '..', '..',  'config.yaml'))

        # Load configuation file
        with open(config_file,'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_type = cfg['planner']['image_type']
        # Initialize network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_folder_path = os.path.join(current_file_path,'../..', 'models')
        
        # Load network model
        model_name = cfg['network']["model_name"]
        model_path = os.path.join(model_folder_path, model_name, "model.pt")
        self.threshold =  cfg["network_threshold"]
        self.model = PushNet()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Normalize input data
        data_path = cfg['data_path']
        data_stats_folder_path = data_path+ '/data_stats'
        self.image_mean = np.load(data_stats_folder_path + '/image_mean.npy')
        self.image_std  = np.load(data_stats_folder_path + '/image_std.npy')
        self.masked_image_mean = np.load(data_stats_folder_path + '/masked_image_mean.npy')
        self.masked_image_std  = np.load(data_stats_folder_path + '/masked_image_std.npy')
    
        self.velocities = fibonacci_sphere(2000)
        self.velocity_mean = np.load(data_stats_folder_path + '/velocity_mean.npy')
        self.velocity_std = np.load(data_stats_folder_path + '/velocity_std.npy')
        
    def is_stable(self, depth_image, contact_point, icr):
        '''
        Determine if a given contact point is stable or not.
        Inputs:
        - depth_image: Uncropped (crude) 2D depth image
        - contact_point: ContactPoint object
        - icr: 2D instantaneous center of rotation vector (x, y)
        Returns:
        - quality: 1D array of shape (1, 1). 1 if stable, 0 if unstable. Scaled between 0 and 1.
        '''
        
        # Calculate quality for a given velocity and contact point
        image_tensor = crop_image(depth_image, contact_point)
        
        if self.image_type == 'masked':
            image_tensor = (image_tensor - self.masked_image_mean) / self.masked_image_std
        else: 
            image_tensor = (image_tensor - self.image_mean) / self.image_std
            
        image_tensor = torch.from_numpy(image_tensor.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        velocity = np.array([icr[1], -icr[0], 1])
        if icr[1] < 0:
            velocity = -velocity
        velocity = velocity / np.linalg.norm(velocity)
        velocity = (velocity - self.velocity_mean) / self.velocity_std
        velocity = torch.from_numpy(velocity.astype(np.float32)).unsqueeze(0)
        
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            velocity = velocity.to(self.device)
            
            output = self.model(image_tensor, velocity)
            quality = torch.nn.Softmax(dim=1)(output)[0,1].cpu()
            if quality < self.threshold:
                quality = torch.Tensor([0])
                # quality = False
                
        # return quality
        return quality.numpy()
    
    def get_push_results(self, depth_image, contact_point, icrs):
        '''
        Get push result for a given icr
        Inputs:
        - depth_image: Uncropped (crude) 2D depth image
        - contact_point: ContactPoint object
        - icrs: 2D instantaneous centers of rotation vector (n, 2)
        Returns:
        - quality: 1D array of shape (1, 1). 1 if stable, 0 if unstable. Scaled between 0 and 1.
        '''
        
        # Calculate quality for a given velocity and contact point
        image_tensor = crop_image(depth_image, contact_point)
        
        if self.image_type == 'masked':
            image_tensor = (image_tensor - self.masked_image_mean) / self.masked_image_std
        else: 
            image_tensor = (image_tensor - self.image_mean) / self.image_std
            
        image_tensor = torch.from_numpy(image_tensor.astype(np.float32))
        
        velocities = []
        for icr in icrs:
            velocity = np.array([icr[1], -icr[0], 1])
            if icr[1] < 0:
                velocity = -velocity
            velocity = velocity / np.linalg.norm(velocity)
            velocity = (velocity - self.velocity_mean) / self.velocity_std
            velocities.append(velocity)
            
        velocities = torch.from_numpy(np.array(velocities).astype(np.float32))
        
        with torch.no_grad():
            
            image_device = image_tensor.to(self.device)
            image_device = image_device.unsqueeze(0)
            images_device = torch.tile(image_device, (len(velocities),1,1,1))
            
            velocities_device = velocities.to(self.device)
            
            outputs = self.model(images_device,velocities_device).cpu()
            results = torch.nn.Softmax(dim=1)(outputs)[:,1]
            qualities = results.detach().numpy()
            
        results = np.where(qualities > self.threshold, 1, 0)
        # return quality
        return results
    
    def check_all_velocities(self, depth_image, contact_point):
        """
        Check the stability for all velocities for a given push contact.
        
        Inputs:
        - depth_image: Uncropped (crude) 2D depth image
        - contact_point: ContactPoint object
    
        Returns:
        - quality: 1D array of shape (n, 1). (n is the number of total velocities.) 1 if stable, 0 if unstable. Scaled between 0 and 1.
        
        """
        
        # Calculate quality for a given velocity and contact point
        image = crop_image(depth_image, contact_point)
        
        if self.image_type == 'masked':
            normalized_image = (image - self.masked_image_mean) / self.masked_image_std
        else: 
            normalized_image = (image - self.image_mean) / self.image_std
            
        normalized_velocities = (self.velocities - self.velocity_mean) / self.velocity_std
        
        with torch.no_grad():
            
            image = normalized_image.astype(np.float32)
            image_device = torch.from_numpy(image).to(self.device)
            image_device = image_device.unsqueeze(0)
            images_device = torch.tile(image_device, (len(normalized_velocities),1,1,1))
            
            velocities = normalized_velocities.astype(np.float32)
            velocities_device = torch.from_numpy(velocities).to(self.device)
            
            outputs = self.model(images_device,velocities_device).cpu()
            results = torch.nn.Softmax(dim=1)(outputs)[:,1]
            qualities = results.detach().numpy()
            
        return qualities 

class DepthImageBasedDeterminator(object):
    def __init__(self, friction_coefficient):
        self.mu = friction_coefficient
        velocities = fibonacci_sphere(2000)
        self.icrs = velocity2icr(velocities)
    def is_stable(self, depth_image, contact_point, icr):
        """
        Determine if a given contact point is stable or not.

        Args:
            depth_image (numpy.ndarray): (H, W) depth image.
            contact_point (ContactPoint): ContactPoint object.
            icr (numpy.ndarray): (2,) instantaneous center of rotation vector (x, y)

        Returns:
            is_stable (bool): True if stable, False otherwise.
        """
        # Depth image input is just for input uniformity
        edge_point_xy = contact_point.edge_xyz[:, :2]
        contact_points = contact_point.contact_points[..., :2]
        contact_noramls = contact_point.contact_normals
        stable_region = StableRegion(
            input_points=edge_point_xy,
            contact_points=contact_points,
            contact_normals=contact_noramls,
            default_mu=self.mu)

        # stable_region.plot_constraints()
        
        # Calculate veolocity for debuffing
        velocity = np.array([icr[1], -icr[0], 1])
        if icr[1] < 0:
            velocity = -velocity
        velocity = velocity / np.linalg.norm(velocity)
        
        # Check the given icr is within the stable region
        quality = stable_region.is_stable_in_local_frame(icr)
        
        return quality
    
    def get_push_results(self, depth_image, contact_point, icrs):
        '''
        Get push result for a given icr
        Inputs:
        - depth_image: Uncropped (crude) 2D depth image
        - contact_point: ContactPoint object
        - icrs: 2D instantaneous centers of rotation vector (n, 2)
        Returns:
        - Result: 1D array of shape (1, 1). 1 if stable, 0 if unstable. 
        '''
        
        results = []
        
        for icr in icrs:
            result = self.is_stable(depth_image, contact_point, icr)
            results.append(result)
            
        return results
    def check_all_velocities(self, depth_image, contact_point):
        """
        Check the stability for all velocities for a given push contact.
        
        Inputs:
        - depth_image: Uncropped (crude) 2D depth image
        - contact_point: ContactPoint object
    
        Returns:
        - quality: 1D array of shape (n, 1). (n is the number of total velocities.) 1 if stable, 0 if unstable.
        
        """
        
        edge_point_xy = contact_point.edge_xyz[:, :2]
        contact_points = contact_point.contact_points[...,:2]
        contact_noramls = contact_point.contact_normals
        stable_region = StableRegion(
            input_points=edge_point_xy,
            contact_points=contact_points,
            contact_normals=contact_noramls,
            default_mu=self.mu)

        qualities = []
        for icr in self.icrs:
            quality = stable_region.is_stable_in_local_frame(icr)
            qualities.append(quality)
            
        # qualities = np.array(qualities,dtype='object')
        return qualities

def get_determinaor(self, metric):
    if metric == 'stable_pushnet':
        return StablePushNetDeterminator()

    if metric == 'depth_image_based':
        return DepthImageBasedDeterminator()


if __name__ == '__main__':
    determinator = get_determinaor('stable_pushnet')
    determinator = get_determinaor('depth_image_based')