import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import yaml
import re
from utils.dataloader_parallel import DataLoaderParallel


# Get current file path
current_file_path = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.abspath(os.path.join(current_file_path, '..', '..',  '..', 'config.yaml'))

# Load configuation file
with open(config_file,'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
FILE_ZERO_PADDING_NUM = cfg['file_zero_padding_num']

# Data Directories
DATA_DIR = cfg["data_path"]
tensor_dir = DATA_DIR + '/tensors'
data_stats_dir = DATA_DIR + '/data_stats'

class PushNetDataset(Dataset):
    def __init__(self, dataset_dir, type='train', image_type='masked',num_debug_samples=0):
        """Dataset class for DexNet.

        Args:
            dataset_dir (str): path to the dataset directory.
            split (str, optional): split of the dataset. Defaults to 'image_wise'.
            type (str, optional): type of the dataset. Defaults to 'train'.
            image_type (str, optional): type of the network input image. Defaults to 'masked'.
            transform (bool, optional): transform the data. Defaults to True.
        """
        # dataset option
        self.type = type
        self.image_type = image_type
        
        # data directory
        self.tensor_dir = os.path.join(dataset_dir, 'tensors')
        self.split_dir  = os.path.join(dataset_dir, 'split')
    
        # load indicies
        indices_file = os.path.join(self.split_dir, self.type + '_indices.npy')
        self.indices = np.load(indices_file)
        
        file_list = os.listdir(self.tensor_dir)
        file_list = [file_name for file_name in file_list if file_name.startswith('image')]
        indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
        indices = np.sort(indices)
        self.max_index = indices[-1]
        
        # Data normalization
        self.velocity_mean = np.load(os.path.join(data_stats_dir, 'velocity_mean.npy'))
        self.velocity_std  = np.load(os.path.join(data_stats_dir, 'velocity_std.npy'))
        
        data_loader_parallel = DataLoaderParallel(self.max_index, self.tensor_dir, FILE_ZERO_PADDING_NUM)
        
        if image_type == 'masked':
            
            self.masked_image_list = data_loader_parallel.load_masked_image_tensor_parallel()
            self.masked_image_mean = np.load(os.path.join(data_stats_dir, 'masked_image_mean.npy'))
            self.masked_image_std  = np.load(os.path.join(data_stats_dir, 'masked_image_std.npy'))
            
        else:
            
            self.image_list = data_loader_parallel.load_image_tensor_parallel()
            self.image_mean = np.load(os.path.join(data_stats_dir, 'image_mean.npy'))
            self.image_std  = np.load(os.path.join(data_stats_dir, 'image_std.npy'))
        
        self.velocity_list = data_loader_parallel.load_velocity_tensor_parallel()
        self.label_list = data_loader_parallel.load_label_tensor_parallel()
        
        
        # Only for confusion
        if num_debug_samples:
            self.indices = self.indices[:num_debug_samples]

    def __len__(self):
        # Returns the length of the dataset. Defaults to 'image_wise' split.
        return len(self.indices)

    def __getitem__(self, idx):
        
        
        idx   = self.indices[idx]
        
        if self.image_type == 'masked':
            
            image = self.masked_image_list[idx]
            image = (image - self.masked_image_mean) / self.masked_image_std
            
        else: 
            
            image = self.image_list[idx]
            image = (image - self.image_mean) / self.image_std
        
        velocity = self.velocity_list[idx]
        velocity = (velocity - self.velocity_mean) / self.velocity_std
        
        label = self.label_list[idx]
        label_onehot = torch.from_numpy(np.eye(2)[int(label)].astype(np.float32)) # one-hot encoding
        
        return image, velocity, label_onehot
        
        
        

def main():
    dataset_dir = DATA_DIR
    dexnet_train_dataset = PushNetDataset(dataset_dir,  debug=False)
    train_dataloader = DataLoader(dexnet_train_dataset, batch_size=64, shuffle=True, num_workers=16)
        
        
    # test data loader
    for i, (image, velocity, label) in enumerate(train_dataloader):

        print(len(train_dataloader))
        print('image.shape:', image.shape)
        print('velocity.shape:', velocity.shape)
        print('label.shape:', label.shape)
        print('image.device:', image.device)
        print('velocity.device:', velocity.device)
        print('label.device:', label.device)
        
        
        image = image.permute(0, 2, 3, 1)
        image_tf = image.reshape(-1, image.shape[2], image.shape[3])
        
        # img_mean = torch.mean(image_tf)
        # # print(f"Image mean:\t{img_mean}\n")
        # print(f"Pose:\t{pose}\n")
        
        # plt.imshow(image_tf)
        # plt.show()
        # break
    
if __name__ == '__main__':
    main()
