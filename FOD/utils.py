import os, errno
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from glob import glob
from PIL import Image
from torchvision import transforms, utils

from FOD.Loss import ScaleAndShiftInvariantLoss, L1_CE_Loss

class ToMask(object):
    """
        Convert a 3 channel RGB image into a 1 channel segmentation mask
    """
    def __init__(self, palette_dictionnary):
        self.nb_classes = len(palette_dictionnary)
        # sort the dictionary of the classes by the sum of rgb value -> to have always background = 0
        # self.converted_dictionnary = {i: v for i, (k, v) in enumerate(sorted(palette_dictionnary.items(), key=lambda item: sum(item[1])))}
        self.palette_dictionnary = palette_dictionnary

    def __call__(self, pil_image):
        # avoid taking the alpha channel
        image_array = np.array(pil_image)[:, :, :3]
        # get only one channel for the output
        output_array = np.zeros(image_array.shape, dtype="int")[:, :, 0]

        for label in self.palette_dictionnary.keys():
            rgb_color = self.palette_dictionnary[label]['color_data']
            mask = (image_array == rgb_color)
            output_array[mask[:, :, 0]] = int(label)

        output_array = torch.from_numpy(output_array).unsqueeze(0).long()
        return output_array


def get_total_paths(path, ext):
    return glob(os.path.join(path, '*'+ext))

def get_splitted_dataset(config, split, dataset_name, path_images, path_segmentation):
    list_files = [os.path.basename(im) for im in path_images]
    np.random.seed(config['General']['seed'])
    np.random.shuffle(list_files)
    if split == 'train':
        selected_files = list_files[:int(len(list_files)*config['Dataset']['splits']['split_train'])]
    elif split == 'val':
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train']):int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val'])]
    else:
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val']):]

    path_images = [os.path.join(config['Dataset']['paths']['path_dataset'],
                                dataset_name,
                                config['Dataset']['paths']['path_images'],
                                im)
                    for im in selected_files]

    path_segmentation = [os.path.join(config['Dataset']['paths']['path_dataset'],
                                      dataset_name,
                                      config['Dataset']['paths']['path_segmentations'],
                                      im)
                        for im in selected_files]
    return path_images, path_segmentation

def get_transforms(config):
    im_size = config['Dataset']['transforms']['resize']
    transform_image = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_seg = transforms.Compose([
        transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.NEAREST),
        ToMask(config['Dataset']['classes']),
    ])
    return transform_image, transform_seg

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
        elif config['General']['loss_segmentation'] == 'l1_ce':
            loss_segmentation = L1_CE_Loss(config['General']['weights_loss_l1_ce'])
    return loss_segmentation

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

def mask2img(config, mask):
    palette_dictionnary = config['Dataset']['classes']
    output_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype="int")

    for label in palette_dictionnary.keys():
        indexes = (mask == int(label)).squeeze(-1)
        output_img[indexes] = palette_dictionnary[label]['color_plot_wandb']

    return output_img
