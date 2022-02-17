import json
from glob import glob
from FOD.Predictor import Predictor

with open('config.json', 'r') as f:
    config = json.load(f)

input_images = glob('input/*.jpg') + glob('input/*.png')
predictor = Predictor(config, input_images)
predictor.run()
