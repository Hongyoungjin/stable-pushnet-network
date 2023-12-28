import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from utils.model import PushNet
from utils.dataloader import PushNetDataset
from utils.utils import fibonacci_sphere, velocity2icr
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import yaml


'''
Plot the network output (velocity sphere in which each velocity is colored by the estimated push success rate)
and the corresponding image.


'''
def model_loop(dataloader,model,velocities):
    predictions, images = [], []
    velocities = torch.from_numpy(velocities.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        
        for image, velocity, label_onehot  in tqdm(dataloader):
            
            image_device = image.to(DEVICE)
            images_tensor = torch.tile(image_device,(len(velocities),1,1,1))
            
            outputs = model(images_tensor,velocities).cpu()
            result = torch.nn.Softmax(dim=1)(outputs)[:,1] # Estimated push success rate.
            
            predictions.append(result.detach().numpy())
            images.append(image[0].permute(1,2,0).numpy())
            
    return np.array(predictions), np.array(images)


def plot_model(dataloader,model,velocities,num_samples):
    # figure configuration
    plot_fig = plt.figure()
    image_fig = plt.figure()
    num_samples_per_edge = int(np.ceil(np.sqrt(num_samples)))
    stability, images = model_loop(dataloader,model,velocities)
    stability = stability.reshape(-1,1000)
    for idx in range(num_samples):
        
        # Plot model output
        ax = plot_fig.add_subplot(num_samples_per_edge,num_samples_per_edge,idx+1,projection='3d')
        ax.set_title(f"Result {idx}")
        ax.view_init(elev = 0,azim = 0)
        ax.set_xlabel(r"$v_x$ [m/s]")
        ax.set_ylabel(r"$v_y$ [m/s]")
        ax.set_zlabel(r"$\omega$ [rad]", rotation=0)
        ax.set_box_aspect((1,2,2))
        ax.grid(False)
        p = ax.scatter(_velocity[:,0], _velocity[:,1], _velocity[:,2], stability[idx], c=stability[idx], cmap="jet", s=100, vmin=0, vmax=1)
        plot_fig.colorbar(p, ax=ax)
        
        # Plot image
        ax = image_fig.add_subplot(num_samples_per_edge,num_samples_per_edge,idx+1)
        ax.set_title(f"Image {idx}")
        ax.imshow(images[idx])
        ax.axis('off')

    plt.show()
    
def load_sampler(dataset):
    
    label_list = dataset.label_list
    indices = dataset.indices
    num_true=0
    for index in indices:
        if label_list[index] == 1:
            num_true += 1
    num_false = len(indices) - num_true
    
    portion_true = num_true/len(indices)
    portion_false = num_false/len(indices)
    weights = [1/portion_true if label_list[index] == 1 else 1/portion_false for index in indices]
    
    sampler = WeightedRandomSampler(weights, len(weights))
    
    return sampler
    
if __name__ == '__main__':
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get current file path
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.abspath(os.path.join(current_file_path, '..', '..',  'config.yaml'))

    # Load configuation file
    with open(config_file,'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    image_type = cfg['planner']['image_type']
    MODEL_NAME = cfg['network']['model_name']
    DATA_DIR = cfg["data_path"]
    num_push_cases = cfg['network_output']['num_pushes']
    data_stats_dir = DATA_DIR + '/data_stats'
    
    _velocity = fibonacci_sphere()
    velocity_mean = np.load(data_stats_dir + "/velocity_mean.npy")
    velocity_std = np.load(data_stats_dir + "/velocity_std.npy")
    velocity_normalized = (_velocity - velocity_mean) / velocity_std
    
    model = PushNet()
    model.to(DEVICE)
    model_path = os.path.abspath(os.path.join(current_file_path, '..', '..',  'models', MODEL_NAME, 'model.pt'))
    model.load_state_dict(torch.load(model_path)) #, map_location=torch.device('cpu')
    
    # Getting features and confusion index
    print("Getting features and confusion index")
    test_dataset = PushNetDataset(DATA_DIR, type='test', image_type=image_type, num_debug_samples = num_push_cases)
    test_sampler = load_sampler(test_dataset)
    dataloader = DataLoader(test_dataset, shuffle=True, num_workers=16)
    plot_model(dataloader,model,velocity_normalized, num_push_cases)
    
    
        