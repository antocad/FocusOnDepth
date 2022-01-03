import os
import random
from glob import glob

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from FOD.utils import get_total_paths, get_splitted_dataset, get_transforms

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(img.to('cpu').float())
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

class AutoFocusDataset(Dataset):
    """
        Dataset class for the AutoFocus Task. Requires for each image, its depth ground-truth and
        segmentation mask
        Args:
            :- config -: json config file
            :- dataset_name -: str
            :- split -: split ['train', 'val', 'test']
    """
    def __init__(self, config, dataset_name, split=None):
        self.split = split
        self.config = config

        path_images = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_images'])
        path_depths = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_depths'])
        path_segmentations = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_segmentations'])

        self.paths_images = get_total_paths(path_images, config['Dataset']['extensions']['ext_images'])
        self.paths_depths = get_total_paths(path_depths, config['Dataset']['extensions']['ext_depths'])
        self.paths_segmentations = get_total_paths(path_segmentations, config['Dataset']['extensions']['ext_segmentations'])

        assert (self.split in ['train', 'test', 'val']), "Invalid split!"
        assert (len(self.paths_images) == len(self.paths_depths)), "Different number of instances between the input and the depth maps"
        assert (len(self.paths_images) == len(self.paths_segmentations)), "Different number of instances between the input and the segmentation maps"
        assert (config['Dataset']['splits']['split_train']+config['Dataset']['splits']['split_test']+config['Dataset']['splits']['split_val'] == 1), "Invalid splits (sum must be equal to 1)"
        # check for segmentation

        # utility func for splitting
        self.paths_images, self.paths_depths, self.paths_segmentations = get_splitted_dataset(config, self.split, dataset_name, self.paths_images, self.paths_depths, self.paths_segmentations)

        # Get the transforms
        self.transform_image, self.transform_depth, self.transform_seg = get_transforms(config)

        # get p_flip from config
        self.p_flip = config['Dataset']['transforms']['p_flip'] if split=='train' else 0
        self.p_crop = config['Dataset']['transforms']['p_crop'] if split=='train' else 0
        self.p_rot = config['Dataset']['transforms']['p_rot'] if split=='train' else 0
        self.resize = config['Dataset']['transforms']['resize']

    def __len__(self):
        """
            Function to get the number of images using the given list of images
        """
        return len(self.paths_images)

    def __getitem__(self, idx):
        """
            Getter function in order to get the triplet of images / depth maps and segmentation masks
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.transform_image(Image.open(self.paths_images[idx]))
        depth = self.transform_depth(Image.open(self.paths_depths[idx]))
        segmentation = self.transform_seg(Image.open(self.paths_segmentations[idx]))
        imgorig = image.clone()

        if random.random() < self.p_flip:
            image = TF.hflip(image)
            depth = TF.hflip(depth)
            segmentation = TF.hflip(segmentation)

        if random.random() < self.p_crop:
            random_size = random.randint(256, self.resize-1)
            max_size = self.resize - random_size
            left = int(random.random()*max_size)
            top = int(random.random()*max_size)
            image = TF.crop(image, top, left, random_size, random_size)
            depth = TF.crop(depth, top, left, random_size, random_size)
            segmentation = TF.crop(segmentation, top, left, random_size, random_size)
            image = transforms.Resize((self.resize, self.resize))(image)
            depth = transforms.Resize((self.resize, self.resize))(depth)
            segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)

        if random.random() < self.p_rot:
            #rotate
            random_angle = random.random()*20 - 10 #[-10 ; 10]
            mask = torch.ones((1,self.resize,self.resize)) #useful for the resize at the end
            mask = TF.rotate(mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            image = TF.rotate(image, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            depth = TF.rotate(depth, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            segmentation = TF.rotate(segmentation, random_angle, interpolation=transforms.InterpolationMode.NEAREST)
            #crop to remove black borders due to the rotation
            left = torch.argmax(mask[:,0,:]).item()
            top = torch.argmax(mask[:,:,0]).item()
            coin = min(left,top)
            size = self.resize - 2*coin
            image = TF.crop(image, coin, coin, size, size)
            depth = TF.crop(depth, coin, coin, size, size)
            segmentation = TF.crop(segmentation, coin, coin, size, size)
            #Resize
            image = transforms.Resize((self.resize, self.resize))(image)
            depth = transforms.Resize((self.resize, self.resize))(depth)
            segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)
        # show([imgorig, image, depth, segmentation])
        # exit(0)
        return image, depth, segmentation
