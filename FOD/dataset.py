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
        self.paths_depth = get_total_paths(path_depths, config['Dataset']['extensions']['ext_depths'])
        self.paths_segmentation = get_total_paths(path_segmentations, config['Dataset']['extensions']['ext_segmentations'])
        
        assert (self.split in ['train', 'test', 'val']), "Invalid split!"
        assert (len(self.path_images) == len(self.path_depths)), "Different number of instances between the input and the depth maps"
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
        
        image = self.transform_image(Image.open(self.paths_images[idx]))
        depth = self.transform_depth(Image.open(self.paths_depths[idx]))
        # to do: segmentation

        return image, depth



