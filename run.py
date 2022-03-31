import json
from glob import glob

from torchvision import transforms
from torch.utils.data import DataLoader

from FOD.Predictor import Predictor
from FOD.dataset import TestDataset

with open('config.json', 'r') as f:
    config = json.load(f)

resize = config['Dataset']['transforms']['resize']
transform_image = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_data = TestDataset(config, 'input/', transform=transform_image)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=config['General']['batch_size'],
                             shuffle=False)

predictor = Predictor(config)
predictor.run(test_dataloader)
