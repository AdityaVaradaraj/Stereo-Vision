# Stereo-Vision
Stereo Vision Calibration, Rectification, SSD-based Disparity and Depth Computation
Requirements:

OpenCV or cv2 (4.5.5)
matplotlib
numpy
tqdm

Instructions:

1) Unzip the dataset so that, inside the directory containing the code "stereo_vision_v2.py", there is a folder "data" which contains 3 folders "curule", "pendulum" and "octagon", each of which contain 2 images "im0.png" and "im1.png" and a "calib.txt"

2) Run the code by typing "python3 stereo_vision_v2.py". Enter the dataset name when prompted. Press "Q" to continue running the code when matplotlib windows pop-up. After finding fundamental matrix, Essential matrix, epipolar lines and after warping, the code might take lot of time (about 4 mins) to run for calculating disparity map. Then depth map will be calculated.

3) You will find these images saved in the directory containing the code.
