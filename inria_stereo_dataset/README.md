Inria 3DMovie Dataset v1
========================

The Inria 3DMovie Dataset contains all the stereo pairs and their annotations
used in our ICCV 2013 paper [1]. Most of this data was extracted from the
"StreetDance 3D" [Giwa and Pasquini, 2010] and "Pina" [Wenders, 2011] stereo
movies. Some of the negative stereo pairs were harvested from Flickr and were
originally shot using the Fuji W3 camera. The dataset includes stereo pairs,
ground truth segmentations, poses and person bounding boxes, and is split into
a training and test parts.

The training set includes:

- 438 annotated poses from 232 stereo pairs
- 520 annotated person bounding boxes from 261 pairs
- 247 negative stereo pairs without any person in them

The test set includes:

- 36 stereo sequences, provided as 2727 x 2 frames
- 686 ground truth person segmentation from 180 stereo pairs
- 149 annotated poses from 193 stereo pairs
- 638 annotated person bounding boxes from the same 193 pairs

All the annotations were produced manually. The stereo pairs are provided as
jpegs. Estimated disparity using [2] is also provided for each stereo pair.

If you use this dataset, please cite:

> [1] K. Alahari, G. Seguin, J. Sivic, I. Laptev  
> Pose Estimation and Segmentation of People in 3D Movies  
> Proceedings of the International Conference on Computer Vision (ICCV), 2013.  
> <http://www.di.ens.fr/willow/research/stereoseg/>

### Quickstart
The `code` folder contains three demo MATLAB scripts, which load and display
sample frames, disparity and the corresponding ground truth.

    >> demo_persondetection
    >> demo_pose
    >> demo_segmentation

### Dataset layout
This dataset is split into several folders, one for each task and train/test
part, each folder containing at least three subdirectories:

- `frames`, which holds the stereo pairs
- `disparity`, which holds the disparity maps
- `labels`, which holds the appropriate annotations

A `visualization` directory is also provided for some of the subdatasets to
show how the ground truth looks out of the box.

Disparity maps are provided as matfiles, the `uv` variable holding the whole
flow computed between the left and the right image. We use the horizontal
component of the flow as disparity, i.e. `uv(:,:,1)`.

#### Segmentation labels

Segmentation labels are MATLAB files containing a 3D array named `det_gt`,
where each layer of the third dimension `det_gt(:,:,i)` is the ground truth
segmentation mask for a single person.

#### Pose estimation labels

We provide a summary MATLAB file which holds a struct array `pos` element, in
which each struct `pos(i)` has an `im` field specifying the image location, a
`pose` field specifying the coordinates of each of the 10 annotated joints, and
for the training set an extra `occluded` field specifying for each joint
whether it is occluded or not.

The 10 joints are:

1. right hip
2. left hip
3. right wrist
4. right elbow
5. right shoulder
6. left shoulder
7. left elbow
8. left wrist
9. neck
10. head top

#### Person detection labels

For the train dataset, we provide a MATLAB file very similar to the one we
provide for pose estimation. It contains a single `pos` element, which is a
struct array in which each struct has an `im` field giving the example left
image filename and `x1`, `y1`, `x2`, `y2` fields specifying the detection
bounding box.

For the test dataset, we provide xml files compatible with the VOC devkit,
as well as a summary txt file, in which each line provides information on a
single bounding box annotation in the form `filename 1.0 x1 y1 x2 y2`.

If you have any questions, please contact Guillaume Seguin
<guillaume.seguin@ens.fr>.

### Acknowledgements

This work is partly supported by the Quaero Programme, funded by OSEO, the
MSR-INRIA laboratory, ERC grant Activia, Google and the EIT ICT Labs.

### References

> [2] A. Ayvaci, M. Raptis, S. Soatto  
> Sparse Occlusion Detection with Optical Flow  
> In International Journal of Computer Vision, 2011

### Copyright

We do not own the copyright of the videos and stereo pairs, and only provide
them for non-commercial research purposes.
