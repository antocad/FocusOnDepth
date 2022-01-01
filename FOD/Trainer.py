import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import cv2
import torch.nn as nn

from tqdm import tqdm
from os import replace
from numpy.core.numeric import Inf
from FOD.utils import get_losses, get_optimizer, create_dir
from FOD.FocusOnDepth import FocusOnDepth

import DPT.util.io
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

class Trainer(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        self.model = FocusOnDepth(
                    (3,resize,resize),
                    patch_size=config['General']['patch_size'],
                    emb_dim=config['General']['emb_dim'],
                    resample_dim=config['General']['resample_dim'],
                    read=config['General']['read'],
                    nhead=config['General']['nhead'],
                    nclasses=len(config['Dataset']['classes']) + 1
        )

        # self.model = DPTDepthModel(
        #     path=config['Dataset']['paths']['model_dpt'],
        #     backbone="vitl16_384",
        #     non_negative=True,
        #     enable_attention_hooks=False,
        # )

        #self.model.half()
        self.model.to(self.device)
        #print(self.model)
        #exit(0)
        self.loss_depth, self.loss_segmentation = get_losses(config)
        self.optimizer = get_optimizer(config, self.model)

    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['General']['epochs']
        if self.config['wandb']['enable']:
            wandb.init(project="FocusOnDepth", entity=self.config['wandb']['username'])
            wandb.config = {
                "learning_rate": self.config['General']['lr'],
                "epochs": epochs,
                "batch_size": self.config['General']['batch_size']
            }
        val_loss = Inf
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            self.model.train()
            for i, (X, Y_depths, Y_segmentations) in tqdm(enumerate(train_dataloader)):
                # get the inputs; data is a list of [inputs, labels]
                # X, Y_depths, Y_segmentations = X.to(self.device).half(), Y_depths.to(self.device).half(), Y_segmentations.to(self.device).half()
                X, Y_depths, Y_segmentations = X.to(self.device), Y_depths.to(self.device), Y_segmentations.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimizer
                output_depths, output_segmentations = self.model(X)
                output_depths = output_depths.squeeze(1)
                Y_depths = Y_depths.squeeze(1) #1xHxW -> HxW
                Y_segmentations = Y_segmentations.squeeze(1) #1xHxW -> HxW

                # get loss
                loss = self.loss_depth(output_depths, Y_depths) + self.loss_segmentation(output_segmentations, Y_segmentations)
                loss.backward()

                # step optimizer
                self.optimizer.step()
                running_loss += loss.item()

                if self.config['wandb']['enable']:
                    wandb.log({"loss": loss.item()})

                if i%50 == 0:
                    print('epoch {} : loss = '.format(epoch+1), running_loss/(50*self.config['General']['batch_size']))
                    running_loss = 0
            new_val_loss = self.run_eval(train_dataloader, val_dataloader)
            if new_val_loss < val_loss:
                self.save_model()
                val_loss = new_val_loss
        print('Finished Training')

    def run_eval(self, train_dataloader, val_dataloader):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- train_dataloader -: torch dataloader
            :- val_dataloader -: torch dataloader
        """
        val_loss = 0.
        validation_samples = ()
        depth_truth_samples = ()
        depth_preds_samples = ()
        segmentation_truth_samples = ()
        segmentation_preds_samples = ()

        self.model.eval()
        with torch.no_grad():
            for i, (X, Y_depths, Y_segmentations) in tqdm(enumerate(val_dataloader)):
                # X, Y_depths, Y_segmentations = X.to(self.device).half(), Y_depths.to(self.device).half(), Y_segmentations.to(self.device).half()
                X, Y_depths, Y_segmentations = X.to(self.device), Y_depths.to(self.device), Y_segmentations.to(self.device)

                output_depths, output_segmentations = self.model(X)
                output_depths = output_depths.squeeze(1)
                Y_depths = Y_depths.squeeze(1)
                Y_segmentations = Y_segmentations.squeeze(1)

                # get loss
                loss = self.loss_depth(output_depths, Y_depths) + self.loss_segmentation(output_segmentations, Y_segmentations)
                val_loss += loss.item()

                #output_segmentations = b, nbClasses, H, W ->  (1, nbClasses, H, W)

                if len(validation_samples) < self.config['wandb']['images_to_show']:
                    validation_samples = (*validation_samples, X[0].unsqueeze(0))
                    depth_truth_samples = (*depth_truth_samples, Y_depths[0].unsqueeze(0).unsqueeze(0))
                    depth_preds_samples = (*depth_preds_samples, output_depths[0].unsqueeze(0).unsqueeze(0))
                    segmentation_truth_samples = (*segmentation_truth_samples, Y_segmentations[0].unsqueeze(0).unsqueeze(0))
                    segmentation_preds_samples = (*segmentation_preds_samples, output_segmentations[0].unsqueeze(0))

            val_loss = val_loss / len(val_dataloader)
            print('val_loss = ', val_loss)

            if self.config['wandb']['enable']:

                wandb.log({"val_loss": val_loss})

                imgs = torch.cat(validation_samples, dim=0).detach().cpu().numpy()
                imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

                tmp = torch.cat(depth_truth_samples, dim=0).detach().cpu().numpy()
                depth_truths = np.repeat(tmp, 3, axis=1)

                tmp = torch.cat(depth_preds_samples, dim=0).detach().cpu().numpy()
                depth_preds = np.repeat(tmp, 3, axis=1)
                depth_preds = (depth_preds - depth_preds.min()) / (depth_preds.max() - depth_preds.min() + 1e-8)

                tmp = torch.cat(segmentation_truth_samples, dim=0).detach().cpu().numpy()
                segmentation_truths = np.repeat(tmp, 3, axis=1).astype('float32')

                #(3, nbClasses, H, W)
                tmp = torch.argmax(torch.cat(segmentation_preds_samples, dim=0), dim=1)
                tmp = tmp.unsqueeze(1).detach().cpu().numpy()
                segmentation_preds = np.repeat(tmp, 3, axis=1)
                segmentation_preds = segmentation_preds.astype('float32')


                print("******************************************************")
                print(imgs.shape, imgs.mean().item(), imgs.max().item(), imgs.min().item())
                print(depth_truths.shape, depth_truths.mean().item(), depth_truths.max().item(), depth_truths.min().item())
                print(depth_preds.shape, depth_preds.mean().item(), depth_preds.max().item(), depth_preds.min().item())
                print(segmentation_truths.shape, segmentation_truths.mean().item(), segmentation_truths.max().item(), segmentation_truths.min().item())
                print(segmentation_preds.shape, segmentation_preds.mean().item(), segmentation_preds.max().item(), segmentation_preds.min().item())
                print("******************************************************")

                imgs = imgs.transpose(0,2,3,1)
                depth_truths = depth_truths.transpose(0,2,3,1)
                depth_preds = depth_preds.transpose(0,2,3,1)
                segmentation_truths = segmentation_truths.transpose(0,2,3,1)
                segmentation_preds = segmentation_preds.transpose(0,2,3,1)

                #val_predictions = np.concatenate((truth, pred), axis=-2).transpose(0,2,3,1)
                #output_dim = (2*int(self.config['wandb']['im_h']), int(self.config['wandb']['im_w']))
                output_dim = (int(self.config['wandb']['im_w']), int(self.config['wandb']['im_h']))

                wandb.log(
                    {"img": [wandb.Image(cv2.resize(im, output_dim), caption='val_{}'.format(i+1)) for i, im in enumerate(imgs)],
                    "imgTruths": [wandb.Image(cv2.resize(im, output_dim), caption='val_truths{}'.format(i+1)) for i, im in enumerate(depth_truths)],
                    "imgPreds": [wandb.Image(cv2.resize(im, output_dim), caption='val_pred{}'.format(i+1)) for i, im in enumerate(depth_preds)],
                    "segTruths": [wandb.Image(cv2.resize(im, output_dim), caption='val_segtruths{}'.format(i+1)) for i, im in enumerate(segmentation_truths)],
                    "segPreds": [wandb.Image(cv2.resize(im, output_dim), caption='val_segpred{}'.format(i+1)) for i, im in enumerate(segmentation_preds)]
                    }
                )
        return val_loss

    def save_model(self):
        path_model = os.path.join(self.config['General']['path_model'], self.model.__class__.__name__)
        create_dir(path_model)
        torch.save(self.model.state_dict(), path_model+'.p')
        print('Model saved at : {}'.format(path_model))
