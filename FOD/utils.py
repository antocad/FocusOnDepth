import os, errno
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from glob import glob
from PIL import Image
from torchvision import transforms, utils

from FOD.Loss import ScaleAndShiftInvariantLoss
from FOD.Custom_augmentation import ToMask

def get_total_paths(path, ext):
    return glob(os.path.join(path, '*'+ext))

def get_splitted_dataset(config, split, dataset_name, path_images, path_depths, path_segmentation):
    list_files = [os.path.basename(im) for im in path_images]
    np.random.seed(config['General']['seed'])
    np.random.shuffle(list_files)
    if split == 'train':
        selected_files = list_files[:int(len(list_files)*config['Dataset']['splits']['split_train'])]
    elif split == 'val':
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train']):int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val'])]
    else:
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val']):]

    path_images = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_images'], im[:-4]+config['Dataset']['extensions']['ext_images']) for im in selected_files]
    path_depths = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_depths'], im[:-4]+config['Dataset']['extensions']['ext_depths']) for im in selected_files]
    path_segmentation = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_segmentations'], im[:-4]+config['Dataset']['extensions']['ext_segmentations']) for im in selected_files]
    return path_images, path_depths, path_segmentation

def get_transforms(config):
    im_size = config['Dataset']['transforms']['resize']
    transform_image = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_depth = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.Grayscale(num_output_channels=1) ,
        transforms.ToTensor()
    ])
    transform_seg = transforms.Compose([
        transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.NEAREST),
        ToMask(config['Dataset']['classes']),
    ])
    return transform_image, transform_depth, transform_seg

def get_losses(config):
    def NoneFunction(a, b):
        return 0
    loss_depth = NoneFunction
    loss_segmentation = NoneFunction
    type = config['General']['type']
    if type == "full" or type=="depth":
        if config['General']['loss_depth'] == 'mse':
            loss_depth = nn.MSELoss()
        elif config['General']['loss_depth'] == 'ssi':
            loss_depth = ScaleAndShiftInvariantLoss()
    if type == "full" or type=="segmentation":
        if config['General']['loss_segmentation'] == 'ce':
            loss_segmentation = nn.CrossEntropyLoss()
    return loss_depth, loss_segmentation

def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# def get_optimizer(config, net):
#     if config['General']['optim'] == 'adam':
#         optimizer = optim.Adam(net.parameters(), lr=config['General']['lr'])
#     elif config['General']['optim'] == 'sgd':
#         optimizer = optim.SGD(net.parameters(), lr=config['General']['lr'], momentum=config['General']['momentum'])
#     return optimizer

def get_optimizer(config, net):
    names = set([name.split('.')[0] for name, _ in net.named_modules()]) - set(['', 'transformer_encoders'])
    params_backbone = net.transformer_encoders.parameters()
    params_scratch = list()
    for name in names:
        params_scratch += list(eval("net."+name).parameters())

    if config['General']['optim'] == 'adam':
        optimizer_backbone = optim.Adam(params_backbone, lr=config['General']['lr_backbone'])
        optimizer_scratch = optim.Adam(params_scratch, lr=config['General']['lr_scratch'])
    elif config['General']['optim'] == 'sgd':
        optimizer_backbone = optim.SGD(params_backbone, lr=config['General']['lr_backbone'], momentum=config['General']['momentum'])
        optimizer_scratch = optim.SGD(params_scratch, lr=config['General']['lr_scratch'], momentum=config['General']['momentum'])
    return optimizer_backbone, optimizer_scratch

def get_schedulers(optimizers):
    return [ReduceLROnPlateau(optimizer) for optimizer in optimizers]
