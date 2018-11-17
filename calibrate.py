import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# Prepare object points
# Number of inside corners in a calibration chessboard
nx = 9
ny = 6

# Accessing calibration images
cal_files = glob.glob('camera_cal/calibration*.jpg')

# Empty arrays for object points and image points
objpoints = [] # 3D point in real world space
imgpoints = [] # 2D points in image space

# Variable for object points -- will be same for each image
obj_temp = np.zeros((nx*ny, 3), np.float32)
obj_temp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Looping through all calibration files
for fname in cal_files:

    # Looking for image points
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If image points found, saving them and corresponding object points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(obj_temp)

# Calculating calibration parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Saving the needed calibration parameters with pickle
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('camera_cal/calibration_pickle.p', 'wb'))
