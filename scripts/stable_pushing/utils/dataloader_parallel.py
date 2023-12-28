import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import yaml
import parmap
import re
import multiprocessing
from itertools import repeat


class DataLoaderParallel(Dataset):
    def __init__(self, max_index, tensor_dir, file_zero_padding_num):
        """Dataset class for DexNet.

        Args:
            dataset_dir (str): path to the dataset directory.
            split (str, optional): split of the dataset. Defaults to 'image_wise'.
            type (str, optional): type of the dataset. Defaults to 'train'.
            transform (bool, optional): transform the data. Defaults to True.
        """
        self.max_index = max_index
        self.tensor_dir = tensor_dir
        self.file_zero_padding_num = file_zero_padding_num
            
    @staticmethod
    def load_image_tensor(idx, tensor_dir, file_zero_padding_num):
        """Load image tensor from file.

        Args:
            idx (`int`): file index. (e.g. 0, 1, 2, ...)

        Returns:
            `np.ndarray`: image tensor. shape = (N, C, H, W)
        """
        
        im_tensor_name = ("%s_%0" + str(file_zero_padding_num) + "d.npy")%("image", idx)
        im_tensor = np.load(os.path.join(tensor_dir, im_tensor_name), allow_pickle=True)
        im_tensor = im_tensor[np.newaxis]
        
        return {idx:im_tensor.astype(np.float32)}
    
    @staticmethod
    def load_masked_image_tensor(idx, tensor_dir, file_zero_padding_num):
        """Load image tensor from file.

        Args:
            idx (`int`): file index. (e.g. 0, 1, 2, ...)

        Returns:
            `np.ndarray`: image tensor. shape = (N, C, H, W)
        """
        
        im_tensor_name = ("%s_%0" + str(file_zero_padding_num) + "d.npy")%("masked_image", idx)
        im_tensor = np.load(os.path.join(tensor_dir, im_tensor_name), allow_pickle=True)
        im_tensor = im_tensor[np.newaxis]
        
        return {idx:im_tensor.astype(np.float32)}

    @staticmethod
    def load_velocity_tensor(idx, tensor_dir, file_zero_padding_num):
        """Load velocity tensor from file.

        Args:
            idx (`int`): file index. (e.g. 0, 1, 2, ...)
        Returns:
            `np.ndarray`: velocity tensor. shape = (N, 3)
        """
        vel_tensor_name = ("%s_%0" + str(file_zero_padding_num) + "d.npy")%("velocity", idx)
        vel_tensor = np.load(os.path.join(tensor_dir, vel_tensor_name), allow_pickle=True)
        return {idx: vel_tensor.astype(np.float32)}
    
    @staticmethod
    def load_label_tensor(idx, tensor_dir, file_zero_padding_num):
        """load label tensor.

        Args:
            idx (`int`): label index. (e.g. 0, 1, 2, ...)

        Returns:
            `np.ndarray`: label tensor. shape = (N,) [1 0 0 1 0 ...]
        """
        label_tensor_name = ("%s_%0" + str(file_zero_padding_num) + "d.npy")%("label", idx)
        label_tensor = np.load(os.path.join(tensor_dir, label_tensor_name), allow_pickle=True)
        return {idx:label_tensor.astype(np.float32)}
    
    def load_image_tensor_parallel(self):
        # Number of CPU cores available
        num_workers = multiprocessing.cpu_count()
        
        # Load all tensors using multiprocessing
        image_idx_list    = parmap.starmap(self.load_image_tensor, 
                                           list(zip(range(self.max_index), 
                                                 repeat(self.tensor_dir), 
                                                 repeat(self.file_zero_padding_num))),
                                           pm_processes = num_workers, pm_chunksize = num_workers, pm_pbar = {'desc':'Loading image tensor...'})
        
        # Sort tensors by index
        image_idx_list =    sorted(image_idx_list, key=lambda x: list(x.keys())[0])
        # Extract tensors from list of dictionaries
        image_list =    [list(image_idx_dict.values())[0]    for image_idx_dict    in image_idx_list]
        
        return image_list
    
    def load_masked_image_tensor_parallel(self):
        # Number of CPU cores available
        num_workers = multiprocessing.cpu_count()
        
        # Load all tensors using multiprocessing
        image_idx_list    = parmap.starmap(self.load_masked_image_tensor, 
                                           list(zip(range(self.max_index), 
                                                 repeat(self.tensor_dir), 
                                                 repeat(self.file_zero_padding_num))),
                                           pm_processes = num_workers, pm_chunksize = num_workers, pm_pbar = {'desc':'Loading masked image tensor...'})
        
        # Sort tensors by index
        image_idx_list =    sorted(image_idx_list, key=lambda x: list(x.keys())[0])
        # Extract tensors from list of dictionaries
        image_list =    [list(image_idx_dict.values())[0]    for image_idx_dict    in image_idx_list]
        
        return image_list
    
    def load_velocity_tensor_parallel(self):
        # Number of CPU cores available
        num_workers = multiprocessing.cpu_count()
        
        # Load all tensors using multiprocessing
        velocity_idx_list = parmap.starmap(self.load_velocity_tensor, 
                                           list(zip(range(self.max_index),
                                                    repeat(self.tensor_dir),
                                                    repeat(self.file_zero_padding_num))),
                                           pm_processes = num_workers, pm_chunksize = num_workers, pm_pbar = {'desc':'Loading velocity tensor...'})
        
        # Sort tensors by index
        velocity_idx_list = sorted(velocity_idx_list, key=lambda x: list(x.keys())[0])
        # Extract tensors from list of dictionaries
        velocity_list = [list(velocity_idx_dict.values())[0] for velocity_idx_dict in velocity_idx_list]
        
        return velocity_list
    
    def load_label_tensor_parallel(self):
        # Number of CPU cores available
        num_workers = multiprocessing.cpu_count()
        
        # Load all tensors using multiprocessing
        label_idx_list    = parmap.starmap(self.load_label_tensor, 
                                        list(zip(range(self.max_index),
                                                 repeat(self.tensor_dir), 
                                                 repeat(self.file_zero_padding_num))),
                                        pm_processes = num_workers, pm_chunksize = num_workers, pm_pbar = {'desc':'Loading label tensor...'})
        
        # Sort tensors by index
        label_idx_list =    sorted(label_idx_list, key=lambda x: list(x.keys())[0])
        # Extract tensors from list of dictionaries
        label_list =    [list(label_idx_dict.values())[0]    for label_idx_dict    in label_idx_list]
        
        return label_list