import scipy.io

file_to_test = '../inria_stereo_dataset_video_segmentation_disparity/inria_stereo_dataset/video_segmentation/disparity/1/00012249.jpg_disparity.mat' 
mat = scipy.io.loadmat(file_to_test)
print(mat['uv'])