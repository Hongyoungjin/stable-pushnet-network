import os
import numpy as np
import random
import re
import yaml


# Get current file path
current_file_path = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.abspath(os.path.join(current_file_path, '..', '..',  'config.yaml'))

# Load configuation file
with open(config_file,'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Data Directories
DATA_DIR = cfg["data_path"]
tensor_dir = DATA_DIR + '/tensors'
save_dir = DATA_DIR + "/split"

os.makedirs(save_dir,exist_ok=True)
file_list = os.listdir(tensor_dir)
file_list = [file for file in file_list if file.endswith('.npy')]

# Image files
file_list_image = [file for file in file_list if file.startswith('image')]

# Extract numbers and delete zero padding from file names 
numbers = [int(re.search(r'(\d+)(\.\d+)?', file).group()) for file in file_list]
maximum_number = np.max(numbers)

indices = [i for i in range(maximum_number)]
random.shuffle(indices)

train_indices = np.array(indices[:int(maximum_number*0.8)])
test_indices = np.array(indices[int(maximum_number*0.8):])
plot_indices  = np.arange(maximum_number)


with open (save_dir + "/train_indices.npy","wb") as f:
    np.save(f,train_indices)
with open (save_dir + "/test_indices.npy","wb") as f:
    np.save(f, test_indices)
with open (save_dir + "/plot_indices.npy","wb") as f:
    np.save(f, test_indices)
    
print("Train indices: ",train_indices)
print("Test indices: ",test_indices)
print("Plot indices: ",plot_indices)