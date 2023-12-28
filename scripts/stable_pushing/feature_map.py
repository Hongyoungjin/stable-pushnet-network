import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.model import PushNet
from utils.dataloader import PushNetDataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
torch.multiprocessing.set_sharing_strategy('file_system')
# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Get current file path
current_file_path = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.abspath(os.path.join(current_file_path, '..', '..',  'config.yaml'))

# Load configuation file
with open(config_file,'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

image_type = cfg['planner']['image_type']

# Data Directories
DATA_DIR = cfg["data_path"]
tensor_dir = DATA_DIR + '/tensors'
save_dir = DATA_DIR + "/split"

MODEL_NAME = cfg['network']['model_name']
NUM_SAMPLES = cfg['feature']['num_data_points']
NUM_SAMPLES = 5000

# Feature extraction loop
def feature_extraction(dataloader, model, feat_model):
    feature_map_size = 0
    TP = []
    FP = []
    FN = []
    TN = []
    confusion = []
    features = np.zeros([1,128])
    pbar = tqdm(dataloader)

    with torch.no_grad():
        for idx, (images, poses, labels) in enumerate(pbar):
            images = images.to(DEVICE)
            poses = poses.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            feature = feat_model(images,poses).cpu()
            features = np.concatenate((features,feature), axis=0)
            output = model(images,poses).cpu()
            
            # confusion
            pred = torch.argmax(output, dim=1)
            labels = torch.argmax(labels, dim=1 )

            # idx - class
            if pred == 1 and labels == 1: # TP
                TP.append(idx)                                                     
                confusion.append(0)
            elif pred == 1 and labels == 0: # FP
                FP.append(idx)
                confusion.append(1)
                
            elif pred == 0 and labels == 1: # FN
                FN.append(idx)
                confusion.append(2)
                
            elif pred == 0 and labels == 0: # TN
                TN.append(idx)
                confusion.append(3)

            feature_map_size +=1
            if feature_map_size == NUM_SAMPLES:
              features = features[1:]
              confusion = np.array(confusion)
              break
              
    return features, TP, FP, FN, TN, confusion

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    normalized = x - np.min(x)
    return normalized / value_range

# NewModel
class NewModel(nn.Module):
    def __init__(self, original_model, *args):
        super().__init__()
        
        self.im_stream = nn.Sequential(*list(original_model.children())[0])
        self.pose_stream = nn.Sequential(*list(original_model.children())[1])
        self.merge_stream = nn.Sequential(*list(original_model.children())[2][:2])

    def forward(self, image, pose):
        image = self.im_stream(image)
        pose = self.pose_stream(pose)
        feature = torch.cat((image, pose), 1)
        return self.merge_stream(feature)
        
if __name__ == '__main__':
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = os.path.abspath(os.path.join(current_file_path, '..', '..',  'models', MODEL_NAME, 'model.pt'))
    
    model = PushNet()
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path))#, map_location=torch.device('cpu')
    
    # Model for feature extraction
    feat_model = NewModel(model)
    
    # Getting features and confusion index
    print("Getting features and confusion index")
    dexnet_dataset = PushNetDataset(DATA_DIR, image_type=image_type, type='test')
    dataloader = DataLoader(dexnet_dataset, shuffle=True, num_workers=16)
    features, TP, FP, FN, TN, confusion  = feature_extraction(dataloader, model, feat_model)
    
    # Dimensionality reduction
    print("Dimensionality reduction")

    # PCA to reduce dimensionality to 50 
    time_start = time.time()
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(features)
    print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

    # t-SNE to reduce dimensionality to 3 
    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=3000)
    tsne_pca_results3d = tsne.fit_transform(pca_result_50)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    tx = scale_to_01_range(tsne_pca_results3d[:,0])
    ty = scale_to_01_range(tsne_pca_results3d[:,1])
    tz = scale_to_01_range(tsne_pca_results3d[:,2])

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111, projection='3d')
    classes = [TP, FP, FN, TN]
    legends = ['TP', 'FP', 'FN', 'TN']
    colors = ['b', 'y', 'r', 'k']

    for color, each_class in enumerate(classes):
        
        indices = each_class
        
        current_tx = np.take(tx,indices)
        current_ty = np.take(ty,indices)
        current_tz = np.take(tz,indices)
        
        color = colors[color]
        
        ax.scatter(current_tx, current_ty, current_tz, c=color) #, label=legends[color]
    ax.legend(legends)
    ax.grid(False)
    plt.show()

    # t-SNE to reduce dimensionality to 2 
    print("t-SNE to reduce dimensionality to 2")
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
    tsne_pca_results2d = tsne.fit_transform(pca_result_50)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    tx = np.expand_dims(scale_to_01_range(tsne_pca_results2d[:,0]), axis=1)
    ty = np.expand_dims(scale_to_01_range(tsne_pca_results2d[:,1]), axis=1)
    confusion_label = np.expand_dims(confusion, axis=1)

    data = np.concatenate((tx, ty), axis=1)    
    # data = np.concatenate((data[1:], confusion_label), axis=1)   
    data = np.concatenate((data, confusion_label), axis=1)   

    df = pd.DataFrame(data, columns=['f_1', 'f_2', 'label'])    
    ax = sns.scatterplot(data = df, x="f_1", y="f_2", hue="label", palette="Set2")
    sns.set(rc = {'figure.figsize':(30,30)})
    ax.grid(False)
    ax.set_aspect('equal')
    plt.show()