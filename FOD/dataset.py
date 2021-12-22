import os
import torch
from PIL import Image

from tqdm import tqdm
from glob import glob


from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from FOD.utils import get_total_paths, get_splitted_dataset, get_transforms

class AutoFocusDataset(Dataset):
    """
        Dataset class for the AutoFocus Task. Requires for each image, its depth ground-truth and
        segmentation mask
        Args:
            :- config -: json config file 
            :- split -: split ['train', 'val', 'test']
    """
    def __init__(self, config, split=None):
        self.split = split
        self.config = config

        self.path_images = get_total_paths(config['Dataset']['paths']['path_images'], config['Dataset']['extensions']['ext_images'])
        self.path_depth = get_total_paths(config['Dataset']['paths']['path_depth'], config['Dataset']['extensions']['ext_depth'])
        self.path_segmentation = get_total_paths(config['Dataset']['paths']['path_segmentation'], config['Dataset']['extensions']['ext_segmentation'])
        
        assert (self.split in ['train', 'test', 'val']), "Invalid split!"
        assert (len(self.path_images) == len(self.path_depth)), "Different number of instances between the input and the depth maps"
        assert (config['Dataset']['splits']['split_train']+config['Dataset']['splits']['split_test']+config['Dataset']['splits']['split_val'] == 1), "Invalid splits (sum must be equal to 1)"
        # check for segmentation

        # utility func for splitting
        self.path_images, self.path_depths, self.path_segmentation = get_splitted_dataset(config, self.split, self.path_images, self.path_depth, self.path_segmentation)

        # Get the transforms
        self.transform_image, self.transform_depth, self.transform_seg = get_transforms(config)

    def __len__(self):
        """
            Function to get the number of images using the given list of images
        """
        return len(self.path_images)
    
    def __getitem__(self, idx):
        """
            Getter function in order to get the predicted keypoints from an example image.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.transform_image(Image.open(self.path_images[idx]))
        depth = self.transform_depth(Image.open(self.path_depths[idx]))
        # to do: segmentation

        return image, depth



