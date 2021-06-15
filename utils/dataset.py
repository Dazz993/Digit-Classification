import os
import numpy as np
import torch
import cv2
import scipy.io as sio
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset

from .feature_extraction.hog import hog

feature_extraction_dict = {
    'hog': hog
}

def get_dataset(path='../data/', feature_extraction=None, **kwargs):
    # define normalization and transforms
    normalize = transforms.Normalize(mean=[0.449, 0.450, 0.450], std=[0.200, 0.199, 0.199])

    if feature_extraction is None:
        train_dataset = SVHNDataset(root=path, split='train', transform=normalize)
        test_dataset = SVHNDataset(root=path, split='test', transform=normalize)
    elif feature_extraction in feature_extraction_dict:
        feature_extraction_func = feature_extraction_dict[feature_extraction]
        train_dataset = SVHNDataset(root=path, split='train', feature_extraction=feature_extraction_func, **kwargs)
        test_dataset = SVHNDataset(root=path, split='test', feature_extraction=feature_extraction_func, **kwargs)
    else:
        raise NotImplementedError

    return train_dataset, test_dataset


class SVHNDataset(VisionDataset):
    def __init__(self, root, split='train', feature_extraction=None, transform=None, target_transform=None, **kwargs):
        super(SVHNDataset, self).__init__(root=root, transform=transform, target_transform=target_transform)
        loaded_mat = sio.loadmat(os.path.join(root, f'{split}_32x32.mat'))
        self.data = loaded_mat['X'].astype(float) / 255
        self.data = self.data.transpose(3, 2, 0, 1)

        # if feature_extraction is not None:
        #     self.data = feature_extraction(self.data, **kwargs)

        if feature_extraction is not None:
            num_orientations = kwargs.get('num_orientations', 9)
            pixels_per_cell = kwargs.get('pixels_per_cell', (4, 4))
            cells_per_block = kwargs.get('cells_per_block', (1, 1))

            features_path = os.path.join(root, f'{split}_{feature_extraction.__name__}_{num_orientations}_{pixels_per_cell[0]}_{cells_per_block[0]}.npy')
            if not os.path.exists(features_path):
                self.data = feature_extraction(self.data, **kwargs)
                np.save(features_path, self.data)
            else:
                self.data = np.load(features_path)

        self.labels = loaded_mat['y'].astype(np.int64).squeeze()
        np.place(self.labels, self.labels == 10, 0)

    def __getitem__(self, index: int):
        img, target = torch.tensor(self.data[index]).float(), int(self.labels[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    a, b = get_dataset(feature_extraction='hog', num_orientations=18)