# Focus On Depth - A single DPT encoder for AutoFocus application

## Abstract


> Recent works have shown that in the real world, humans
rely on the image obtained by their left and right eyes in or-
der to estimate depths of surrounding objects. Thus, depth
estimation is a classic task in computer vision, which is of
great significance for many applications such as augmented
reality, target tracking and autonomous driving. We firstly
summarize the deep learning models for monocular depth
estimation. Secondly, we will implement a recent Vision
Transformers based architecture for this task. We will seek
to improve it by adding a segmentation head in order to
perform multi-task learning using a customly built dataset.
Thirdly, we will implement our model for in-the-wild im-
ages (i.e. with no control on the environment, the distance
and size of objects of interests, and their physical properties
(rotation, dynamics, etc.)) for Auto-focus application on
humans and will give qualitative comparison across other
methods.

## TO DOs

- [x] Make the training script work
- [x] Create the dataset with person segmentation and depth estimation
- [x] Create a strong codebase for training
- [x] Add data augmentations
- [x] Make the code modulable with timm and the list of hooks
- [x] Add an option to select whether we want depth only or segmentation only or both
- [ ] Make 2 optimizers?
- [ ] Create a strong code base for inference

## Requirements

Run: ``` pip install -r requirements.txt ```

### Training

#### Build the dataset

Our model is trained on a combination of
+ [inria movie 3d dataset](https://www.di.ens.fr/willow/research/stereoseg/).
+ [NYU2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
+ [PoseTrack](https://posetrack.net/)

##### Inria 3d Movie Dataset

1. Download the disparity video frames directly from [here](https://www.di.ens.fr/willow/research/stereoseg/dataset/inria_stereo_dataset_video_segmentation_disparity.tar.gz).
2. Get the segmentation masks from ...

##### NYU2 Dataset

1. Download the labeled dataset directly from [here](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat).
