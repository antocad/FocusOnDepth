# Focus On Depth - A single DPT encoder for AutoFocus application

## Requirements

Run: ``` pip install -r requirements.txt ```

### Training

#### Build the dataset

Our model is trained on a combination of 
+ [inria movie 3d dataset](https://www.di.ens.fr/willow/research/stereoseg/). 
+ [NYU2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

##### Inria 3d Movie Dataset

1. Download the disparity video frames directly from [here](https://www.di.ens.fr/willow/research/stereoseg/dataset/inria_stereo_dataset_video_segmentation_disparity.tar.gz).
2. Get the segmentation masks from ...

##### NYU2 Dataset

1. Download the labeled dataset directly from [here](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat).