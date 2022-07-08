import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter

from PIL import Image

from FOD.FocusOnDepth import FocusOnDepth
from FOD.utils import create_dir
from FOD.dataset import show


class Predictor(object):
    def __init__(self, config, input_images):
        self.input_images = input_images
        self.config = config
        self.type = self.config['General']['type']

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        self.model = FocusOnDepth(
                    image_size  =   (3,resize,resize),
                    emb_dim     =   config['General']['emb_dim'],
                    resample_dim=   config['General']['resample_dim'],
                    read        =   config['General']['read'],
                    nclasses    =   len(config['Dataset']['classes']) + 1,
                    hooks       =   config['General']['hooks'],
                    model_timm  =   config['General']['model_timm'],
                    type        =   self.type,
                    patch_size  =   config['General']['patch_size'],
        )
        path_model = os.path.join(config['General']['path_model'], 'FocusOnDepth_{}.p'.format(config['General']['model_timm']))
        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)['model_state_dict']
        )
        self.model.eval()
        self.transform_image = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.output_dir = self.config['General']['path_predicted_images']
        create_dir(self.output_dir)

    def run(self):
        with torch.no_grad():
            for images in self.input_images:
                pil_im = Image.open(images)
                original_size = pil_im.size

                tensor_im = self.transform_image(pil_im).unsqueeze(0)
                output_depth, output_segmentation = self.model(tensor_im)
                output_depth = 1-output_depth

                output_segmentation = transforms.ToPILImage()(output_segmentation.squeeze(0).argmax(dim=0).float()).resize(original_size, resample=Image.NEAREST)
                output_depth = transforms.ToPILImage()(output_depth.squeeze(0).float()).resize(original_size, resample=Image.BICUBIC)

                path_dir_segmentation = os.path.join(self.output_dir, 'segmentations')
                path_dir_depths = os.path.join(self.output_dir, 'depths')
                create_dir(path_dir_segmentation)
                output_segmentation.save(os.path.join(path_dir_segmentation, os.path.basename(images)))

                path_dir_depths = os.path.join(self.output_dir, 'depths')
                create_dir(path_dir_depths)
                output_depth.save(os.path.join(path_dir_depths, os.path.basename(images)))

                ## TO DO: Apply AutoFocus

                # output_depth = np.array(output_depth)
                # output_segmentation = np.array(output_segmentation)

                # mask_person = (output_segmentation != 0)
                # depth_person = output_depth*mask_person
                # mean_depth_person = np.mean(depth_person[depth_person != 0])
                # std_depth_person = np.std(depth_person[depth_person != 0])

                # #print(mean_depth_person, std_depth_person)

                # mask_total = (depth_person >= mean_depth_person-2*std_depth_person)
                # mask_total = np.repeat(mask_total[:, :, np.newaxis], 3, axis=-1)
                # region_to_blur = np.ones(np_im.shape)*(1-mask_total)

                # #region_not_to_blur = np.zeros(np_im.shape) + np_im*(mask_total)
                # region_not_to_blur = np_im
                # blurred = cv2.blur(region_to_blur, (10, 10))

                # #final_image = blurred + region_not_to_blur
                # final_image = cv2.addWeighted(region_not_to_blur.astype(np.uint8), 0.5, blurred.astype(np.uint8), 0.5, 0)
                # final_image = Image.fromarray((final_image).astype(np.uint8))
                # final_image.save(os.path.join(self.output_dir, os.path.basename(images)))
