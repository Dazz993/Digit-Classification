import os
import time
import torch
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models.resnet_for_feature_analysis import resnet
from utils.dataset import get_dataset
from tqdm import tqdm

'''
Steps to visualize the features of intermediate layers
1. get the feature map of the given intermediate layer
2. reduce the dimension of the feature to 2 by PCA or t-SNE
3. visualize the results
'''

def reduce_dimension_by_pca(features:np.ndarray, n_components=2):
    features = features.reshape(features.shape[0], -1)
    return PCA(n_components=n_components).fit_transform(features)

def reduce_dimension_by_tsne(features:np.ndarray, n_components=2):
    features = features.reshape(features.shape[0], -1)
    return TSNE(n_components=n_components).fit_transform(features)


def visualize_2D(data, labels, name=''):
    assert data.ndim == 2 and labels.ndim == 1
    assert data.shape[0] == len(labels) and data.shape[1] == 2
    plt.cla()
    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='o', s=25, cmap=plt.cm.Set1)
    plt.savefig(f'results/feature_analysis/{name}_{time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())}.png')


if __name__ == '__main__':
    # Load datasets
    n_points = 1000
    _, test_dataset = get_dataset(path='./data/')
    input = torch.stack([test_dataset[i][0] for i in range(n_points)])
    labels = [test_dataset[i][1] for i in range(n_points)]

    # Load vgg model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet(layers=50, num_classes=10).to(device)
    model.eval()

    checkpoing_root = 'states/ResNet50_0.01__2021_06_18_08_21_16'
    checkpoint_lists = [
        os.path.join(checkpoing_root, 'checkpoint_ResNet_0.tar'),
        os.path.join(checkpoing_root, 'checkpoint_ResNet_5.tar'),
        os.path.join(checkpoing_root, 'checkpoint_ResNet_15.tar')
    ]
    name_list = [0, 5, 15]

    fig, axes = plt.subplots(3, 4, figsize=(14, 12), sharex=True, sharey=True)
    for i, checkpoint_path in enumerate(checkpoint_lists):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        features = []
        with torch.no_grad():
            input = input.to(device)
            output, intermediate_feature_1, intermediate_feature_2, intermediate_feature_3, intermediate_feature_4 = model(input)
            features.append(intermediate_feature_1.detach().cpu().numpy())
            features.append(intermediate_feature_2.detach().cpu().numpy())
            features.append(intermediate_feature_3.detach().cpu().numpy())
            features.append(intermediate_feature_4.detach().cpu().numpy())

        for j, feature in tqdm(enumerate(features)):
            reduced_features = reduce_dimension_by_tsne(feature, n_components=2)
            axes[i, j].scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, marker='o', s=25, cmap=plt.cm.Set1)
            axes[i, j].axis('off')
            axes[i, j].set_title(f"Layer {j}, epoch {name_list[i]}")

    plt.savefig('docs/tsne.png')