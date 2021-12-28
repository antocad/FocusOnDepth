import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from FOD.utils import get_loss, get_optimizer
from FOD.FocusOnDepth import FocusOnDepth

class Trainer(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        self.model = FocusOnDepth(
                    (3,config['Dataset']['transforms']['resize'],config['Dataset']['transforms']['resize']),
                    patch_size=config['General']['patch_size'],
                    emb_dim=config['General']['emb_dim'],
                    resample_dim=config['General']['resample_dim'],
                    read=config['General']['read'],
                    nhead=config['General']['nhead']
        )

        self.model.to(self.device)

        self.loss = get_loss(config)
        self.optimizer = get_optimizer(config, self.model)

    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['General']['epochs']
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            self.model.train()
            for i, (x, y_depth) in tqdm(enumerate(train_dataloader)):
                # get the inputs; data is a list of [inputs, labels]
                x, y_depth = x.to(self.device), y_depth.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()


                # forward + backward + optimizer
                outputs_depth = self.model(x)

                #print(i, x.max(), y_depth.max(), outputs_depth.max())
                #print(outputs_depth.sum())

                # get loss
                if self.config['General']['loss'] == 'ssi':
                    outputs_depth = outputs_depth.squeeze(1)
                    y_depth = y_depth.squeeze(1)
                    mask = y_depth > 0
                    y_depth[mask] = (y_depth[mask] - y_depth[mask].min()) / \
                                    (y_depth[mask].max() - y_depth[mask].min())

                    y_depth[mask] = 10. / y_depth[mask]
                    y_depth[~mask] = 0.

                    loss = self.loss(outputs_depth, y_depth, mask)
                else:
                    loss = self.loss(outputs_depth, y_depth)
                loss.backward()

                # step optimizer
                self.optimizer.step()
                running_loss += loss.item()

                #plt.imshow(x[0].permute(1,2,0).detach().numpy())
                #plt.show()
                #plt.imshow(y_depth[0].squeeze().detach().numpy())
                #plt.show()
                #plt.imshow(outputs_depth[0].squeeze().detach().numpy())
                #plt.show()
                #break

            print('epoch {} : loss = '.format(epoch+1), running_loss)

        print('Finished Training')
