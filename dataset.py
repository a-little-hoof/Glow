import torch
import torchvision
from torchvision import transforms
import numpy as np
import imageio as io
from skimage.transform import resize as skresize
import os
import glob

class AFHQDataset(torchvision.datasets.VisionDataset):

    def __init__(self, train=True, input_shape=[3,128,128], num_bits=0, data_root_path='/gpfs/share/home/2301111469/yifei/afhq/train/cat', **kwargs):

        if train == True:   split = 'train'
        else:               split = 'test'

        transform = [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize(input_shape[1:], antialias=True),
        ]
        if num_bits:
            transform.append(
                transforms.Lambda( lambda x : discretize(x, num_bits) )
            )
    
        transform.append(transforms.Lambda(lambda x: x - 0.5))
        transform = transforms.Compose(transform)

        super(AFHQDataset, self).__init__(root=data_root_path, transform=transform, target_transform=None)

        if split == 'train':
            self.image_filenames = sorted(glob.glob(
                os.path.join(self.root, '*.jpg' )
            ))[:-500]
        elif split == 'test':
            self.image_filenames = sorted(glob.glob(
                os.path.join(self.root, '*.jpg' )
            ))[-500:]

        print(f"transorm: {transform}")

    def __getitem__(self, index: int):

        X = io.imread(self.image_filenames[index])
        if self.transform is not None:
            X = self.transform(X)
        
        return X

    def __len__(self):
        return len(self.image_filenames)