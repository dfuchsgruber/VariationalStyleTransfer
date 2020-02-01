import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import random
import pickle

import os

from PIL import Image


def list_images(directory, extensions=('.png', '.jpeg', '.jpg')):
    """ Lists all images in a directory. """
    return [os.path.join(dirpath, filename) for dirpath, dirname, filenames in os.walk(directory) for filename in filenames if any(filename.endswith(extension) for extension in extensions)]

def filter_images(paths, bad_dirs, filter_file, filter_prefix):
    with open(filter_file, "rb") as file:
        good_paths = pickle.load(file)
        good_paths_set = set([os.path.normpath(os.path.join(filter_prefix,path)) for path in good_paths])

        filtered_paths = [path for path in paths if not any(bad_dir in path for bad_dir in bad_dirs)]
        filtered_paths_set = set([os.path.normpath(path) for path in filtered_paths])

        filtered_paths = filtered_paths_set.intersection(good_paths_set)

    return list(filtered_paths)

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

    def __init__(self, paths, resolution=64, random_cropping=True, random_flipping=True, vgg_normalization=False):
        """ Initializes the dataset.
        
        Parameters:
        -----------
        paths : list
            A list of images that form the dataset.
        resolution : int
            Resizes (and random crops for non-square images) to a image of size resolution x resolution
        random_cropping : bool
            If True, a random region will be cropped randomly, else it will be cropped from the center.
        random_flipping : bool
            If True, the image may be flipped horizontally.
        vgg_normalization : bool
            If True, the images are normalized according to the pre-trained vgg networks.
        """
        self.paths = paths
        transformations = [transforms.Resize(resolution)]
        if random_cropping:
            transformations.append(transforms.RandomCrop(resolution))
        else:
            transformations.append(transforms.CenterCrop(resolution))
        if random_flipping:
            transformations.append(transforms.RandomHorizontalFlip())
        transformations.append(transforms.ToTensor())
        if vgg_normalization:
            transformations.append(vgg_normalization)
        self.transformations=transforms.Compose(transformations)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB') # Use a RGB instead of an RGBA image
        image = self.transformations(image)

        return image, self.paths[idx]

    def __len__(self):
        return len(self.paths)

def load_dataset_from_list(files, resolution=64, seed=1337):
    """ Loads some content images. """
    random.seed(seed)
    random.shuffle(files)
    return ImageDataset(files, resolution=resolution, random_cropping=False, random_flipping=False)

def load_dataset(directory, resolution=64, seed=1337, random_transformations=False):
    """ Loads some debug content images. """
    random.seed(seed)
    jpgs = list(map(str, list(Path(directory).rglob("*.jpg"))))
    random.shuffle(jpgs)
    return ImageDataset(jpgs, resolution=resolution, random_cropping=random_transformations, random_flipping=random_transformations)

def load_debug_dataset(directory, resolution=64, number_instances=100000000, random_transformations=False):
    """ Loads some debug style images. """
    return ImageDataset(list_images(directory)[:number_instances], resolution=resolution, random_cropping=random_transformations, random_flipping=random_transformations)

def resize_images_offline(directory, output_dir, resolution=96):
    """ Resizes all images in directory to a new resolution and saves them in output_dir """
    images = list(map(str, list(Path(directory).rglob("*.jpg"))))
    resizeOp = transforms.Resize((resolution, resolution))
    
    for idx, path in enumerate(images):
        image = Image.open(path).convert('RGB') # Use a RGB instead of an RGBA image
        image = resizeOp(image)
        _, filename = os.path.split(path)
        savepath = os.path.join(output_dir, filename)
        image.save(savepath)
        
        if idx % 1000 == 0:
            print(f"Progress: ({idx}/{len(images)})")


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

    def reset(self):
        """ Resets iterator states for both dataset. """
        self.dataset_content_iterator = iter(self.dataset_content)
        self.dataset_style_iterator = iter(self.dataset_style)


class DatasetTripletIterator:
    """ Iterator that endlessly yields triplets of content_image, content_image and style_image. """

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
            content1 = next(self.dataset_content_iterator)
        except StopIteration:
            self.dataset_content_iterator = iter(self.dataset_content)
            content1 = next(self.dataset_content_iterator)
        try:
            content2 = next(self.dataset_content_iterator)
        except StopIteration:
            self.dataset_content_iterator = iter(self.dataset_content)
            content2 = next(self.dataset_content_iterator)
        try:
            style = next(self.dataset_style_iterator)
        except:
            self.dataset_style_iterator = iter(self.dataset_style)
            style = next(self.dataset_style_iterator)
        return content1, content2, style

    def reset(self):
        """ Resets iterator states for both dataset. """
        self.dataset_content_iterator = iter(self.dataset_content)
        self.dataset_style_iterator = iter(self.dataset_style)


class DatasetZippedTripletIterator:
    """ Iterator that yields a "zipped" version content and style images:
    
    For N content and M style images, the iterator returns a batch assembled
    of all N*M different combinations of content images.
    """

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
        
        content, content_paths = content
        style, style_paths = style

        N = content.size(0)
        M = style.size(0)
        content_labels = torch.arange(N).unsqueeze(1).repeat((1, M)).view(-1)
        style_labels = torch.arange(M).repeat(N)
        yield (content[content_labels], content_paths[content_labels]), (style[style_labels], style_paths[style_labels]), content_labels, style_labels





