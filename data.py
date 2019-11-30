import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import random

import os

from PIL import Image


def list_images(directory, extensions=('.png', '.jpeg', '.jpg')):
    """ Lists all images in a directory. """
    return [os.path.join(directory,filename) for filename in os.listdir(directory) if any(filename.endswith(extension) for extension in extensions)]

# VGG19 was trained on images that were normalized with these values
vgg_normalization_mean = np.array([0.485, 0.456, 0.406])
vgg_normalization_std = np.array([0.229, 0.224, 0.225])

vgg_normalization = transforms.Normalize(vgg_normalization_mean, vgg_normalization_std, inplace=False)

def vgg_normalization_undo(image):
    """ Undoes the vgg19 normalization. 
    
    Parameters:
    -----------
    image : torch.Tensor, shape [B, 3, H, W]
        An image that was normalized to be trained on vgg19.

    Returns:
    --------
    recovered : torch.Tensor, shape [B, 3, H, W]
        An image where then mean and std of the vgg19 model were reapplied.
    """
    return (image * vgg_normalization_std.reshape((1, 3, 1, 1))) + vgg_normalization_mean.reshape((1, 3, 1, 1))



class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, paths, resolution=64):
        """ Initializes the dataset.
        
        Parameters:
        -----------
        paths : list
            A list of images that form the dataset.
        resolution : int
            Resizes (and random crops for non-square images) to a image of size resolution x resolution
        """
        self.paths = paths
        self.transformations=transforms.Compose([
            		         transforms.Resize(resolution),
            		         transforms.RandomCrop(resolution),
                             transforms.RandomHorizontalFlip(),
            		         transforms.ToTensor(),
                             vgg_normalization,])

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB') # Use a RGB instead of an RGBA image
        image = self.transformations(image)

        return image, self.paths[idx]

    def __len__(self):
        return len(self.paths)


def load_debug_content_dataset(resolution=64):
    """ Loads some debug content images. """
    random.seed(1337)
    jpgs = list(map(str, list(Path("./dataset/content").rglob("*.jpg"))))
    random.shuffle(jpgs)
    return ImageDataset(jpgs[:1000], resolution=resolution)

def load_debug_style_dataset(resolution=64):
    """ Loads some debug style images. """
    return ImageDataset(list_images('dataset/debug/style'), resolution=resolution)

class DatasetPairIterator:
    """ Iterator that endlessly yields pairs of images. """

    def __init__(self, dataset_content, dataset_style):
        """ Initializes the pairwise dataset iterator.
        
        Parameters:
        -----------
        dataset_content : iterable
            An iterable for content images.
        dataset_style : iterable
            An iterable for style images.
        """
        self.dataset_content = dataset_content
        self.dataset_style = dataset_style
        self.dataset_content_iterator = iter(self.dataset_content)
        self.dataset_style_iterator = iter(self.dataset_style)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            content = next(self.dataset_content_iterator)
        except StopIteration:
            self.dataset_content_iterator = iter(self.dataset_content)
            content = next(self.dataset_content_iterator)
        try:
            style = next(self.dataset_style_iterator)
        except:
            self.dataset_style_iterator = iter(self.dataset_style)
            style = next(self.dataset_style_iterator)
        return content, style
