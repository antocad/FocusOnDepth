import json
import numpy as np
from torch.utils.data import DataLoader

from FOD.Trainer import Trainer
from FOD.dataset import AutoFocusDataset

with open('config.json', 'r') as f:
    config = json.load(f)

np.random.seed(config['General']['seed'])

train_data = AutoFocusDataset(config, 'train')
train_dataloader = DataLoader(train_data, batch_size=config['General']['batch_size'], shuffle=True)

val_data = AutoFocusDataset(config, 'val')
val_dataloader = DataLoader(val_data, batch_size=config['General']['batch_size'], shuffle=True)

trainer = Trainer(config)
trainer.train(train_dataloader, val_dataloader)