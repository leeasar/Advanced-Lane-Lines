import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
import cv2
#from moviepy.editor import VideoFileClip


DEBUG = True
# unpickle
try:
    dist_pickle = pickle.load(open('camera_cal/calibration_pickle.p', 'rb'))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
except:
    print("Calibration pickle file does not exist or is corrupted")

# Correct for distortion
test_location = 'test_images'
test_files = ['test6.jpg', 'test5.jpg', 'test4.jpg', 'test1.jpg', 'test3.jpg', 'test2.jpg', 'straight_lines2.jpg', 'straight_lines1.jpg']
#os.listdir(test_location)

src = np.float32([[588, 460], [697, 460], [1030, 710], [275, 710]])
dst = np.float32([[270, 0], [990, 0], [990, 710], [270, 710]])
M = cv2.getPerspectiveTransform(src, dst)
# Minv

def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    if DEBUG:
        mpimg.imsave(test_location + "/undist_" + file, undist)

    return undist

# ----- Applying multiple thresholding techniques to make lane lines more distinct -----
def hls_select(img, channel, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,channel]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def rgb_select(img, channel, thresh=(0, 255)):
    color_channel = img[:,:,channel]
    binary_output = np.zeros_like(color_channel)
    binary_output[(color_channel > thresh[0]) & (color_channel <= thresh[1])] = 1
    return binary_output

def sobel_thresh(img, direction='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if direction == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobel = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary

def sobel_mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    scaled = np.uint8(255*mag/np.max(mag))
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    return binary_output

def sobel_dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    arc = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(arc)
    binary_output[(arc >= thresh[0]) & (arc <= thresh[1])] = 1
    return binary_output

def thresholding(img):
    hls_bin = hls_select(img, 2, (140, 255))
    sobel_mag_bin = sobel_mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100))
    sobel_bin = sobel_thresh(img, 'x', (19, 200))
    sobel_dir_bin = sobel_dir_thresh(img, 15, thresh=(0.7, 1.3))
    red_bin = rgb_select(img, 0, (120, 255))


    combined_binary = np.zeros_like(hls_bin) #fix this should be more generic
    combined_binary[((red_bin == 1) & (hls_bin == 1)) | (sobel_bin == 1)] = 1
    combined_binary = cv2.GaussianBlur(combined_binary, (7,7), 0)

    if DEBUG:
        mpimg.imsave(test_location + "/thresh_" + file, combined_binary)

    return combined_binary

def warp(img, M, flags=cv2.INTER_LINEAR):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    if DEBUG:
        mpimg.imsave(test_location + "/warp_" + file, warped)

    return warped

for file in test_files:
    img = mpimg.imread(test_location + "/" + file)
    img = undistort(img, mtx, dist)
    img = thresholding(img)
    img = warp(img, M)
