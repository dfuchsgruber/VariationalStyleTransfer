import torch
import torch.utils.data
import torchvision.transforms as transforms

import os

from PIL import Image


def list_images(directory, extensions=('.png', '.jpeg', '.jpg')):
    """ Lists all images in a directory. """
    return [os.path.join(directory,filename) for filename in os.listdir(directory) if any(filename.endswith(extension) for extension in extensions)]

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
            		         transforms.ToTensor()])

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB') # Use a RGB instead of an RGBA image
        image = self.transformations(image)

        return image, self.paths[idx]

    def __len__(self):
        return len(self.paths)


def load_debug_content_dataset(resolution=64):
    """ Loads some debug content images. """
    return ImageDataset(list_images('dataset/debug/content'), resolution=resolution)

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
