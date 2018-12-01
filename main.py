import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
import cv2
import sys
from thresholding_uti import hls_select, rgb_select, sobel_thresh, sobel_mag_thresh, sobel_dir_thresh, canny_thresh
from Line import Line
from RoadVisualizer import RoadVisualizer
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Flip this switch to alternate with normal mode ("False") and debug mode that
# uses test files saves intermediate steps of the pipeline as image files
DEBUG = False

# Retrieving camera calibration from a pickle
try:
    dist_pickle = pickle.load(open('camera_cal/calibration_pickle.p', 'rb'))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
except:
    print("Calibration pickle file does not exist or is corrupted")

# Location of input and output directories
test_location = 'test_images'
output_video_location = 'output_videos'
output_image_location = 'output_images'
test_files = os.listdir(test_location)

# Parameters to project a trapezoid-shape area in original images into a rectangle
# to make lane lines parallel to each other
src = np.float32([[588, 460], [697, 460], [1030, 710], [275, 710]])
dst = np.float32([[270, 0], [990, 0], [990, 710], [270, 710]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)


def undistort(img, mtx, dist):
    """ Undistort an image according to given parameters.

    Args:
        img (RGB image): The original image.
        mtx (array): Undistortion parameters.
        dist (array): Undistortion parameters.

    Returns: The undistorted image.

    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    if DEBUG:
        mpimg.imsave(output_image_location + "/undist_" + file, undist)

    return undist

def thresholding(img):
    """ Use a combination of thresholding techniques to highlight lane lines
    in the image.

    Args:
        img (RGB image): The input image.

    Returns:
        The binary image that highlights the lane lines.
    """

    # Running multiple thresholding techniques
    hls_bin = hls_select(img, 2, (95, 255))
    sobel_mag_bin = sobel_mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100))
    sobel_bin = sobel_thresh(img, 'x', (40, 200))
    sobel_bin_y = sobel_thresh(img, 'y', (55, 200))
    sobel_dir_bin = sobel_dir_thresh(img, 15, thresh=(0.7, 1.3))
    red_bin = rgb_select(img, 0, (50, 255))
    green_bin = rgb_select(img, 1, (85,255))
    canny_bin = canny_thresh(img, (80, 160))

    # Creating the final threshold image by a combination of red channel,
    # saturation channel, and horizontal sobel threshold
    combined_binary = np.zeros_like(hls_bin)
    combined_binary[((red_bin == 1) & (hls_bin == 1)) | ((red_bin == 1) & (sobel_bin == 1))] = 1

    if DEBUG:
        color_img = np.dstack((combined_binary*255, combined_binary*255, combined_binary*255))
        mpimg.imsave(output_image_location + "/thresh_" + file, color_img)

    return combined_binary

def warp(img, M, flags=cv2.INTER_LINEAR):
    """ Project a trapezoid-shape area in an image into a rectangle to make the
    lane lines parallel with each other ("a bird's eye view").

    Args:
        img (RBG or binary image): The input image.
        M (matrix): The transformation matrix.
        flags: The OpenCV2 flags for transformation.

    Returns:
        The warped output image.

    """
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    if DEBUG:
        color_img = np.dstack((warped*255, warped*255, warped*255))
        mpimg.imsave(output_image_location + "/warp_" + file, color_img)

    return warped


def image_to_lane_visualization(img):
    """ An image processing pipeline that through multiple steps creates lane
    visualization on the top of an original road image. Is used by moviepy fl_image
    function to do this visualization for every frame of a video.

    Args:
        img (RGB image): The input image.

    Returns:
        The output image with lane line and road information embedded.
    """
    undist = undistort(img, mtx, dist)
    thresh = thresholding(undist)
    warped = warp(thresh, M)
    left_line.add_new_image(warped) # Global variable because fl_image does not allow passing arguments
    right_line.add_new_image(warped) # Global variable because fl_image does not allow passing arguments
    visualizer = RoadVisualizer([0, 255, 0], Minv)
    visualizer.set_lane_lines(left_line, right_line)
    output = visualizer.visualize(img)

    return output

# Main program -- production mode
if DEBUG == False:

    if len(sys.argv) == 1:
        print("The program main.py requires one video file as an argument")
    else:
        # Read a video file passed by sys.argv
        fname = sys.argv[1]
        if len(sys.argv) == 4:
            start = np.int_(sys.argv[2])
            end = np.int_(sys.argv[3])
            input_clip = VideoFileClip(fname).subclip(start, end)
        else: input_clip = VideoFileClip(fname)
        output_file = output_video_location + "/" + fname
        left_line = Line('left')
        right_line = Line('right')
        # Frame by frame visualization of lane line and road information
        output_clip = input_clip.fl_image(image_to_lane_visualization)
        # Write the output vide in a file
        output_clip.write_videofile(output_file, audio=False)

# Main program -- DEBUG mode
if DEBUG:
    # Go through the test files and output the intermediate pipeline steps into files
    for file in test_files:
        img = mpimg.imread(test_location + "/" + file)
        undist = undistort(img, mtx, dist)
        thresh = thresholding(undist)
        warped = warp(thresh, M)
        left_line = Line('left')
        right_line = Line('right')
        left_line.add_new_image(warped)
        right_line.add_new_image(warped)
        visualizer = RoadVisualizer([0, 255, 0], Minv)
        visualizer.set_lane_lines(left_line, right_line)
        final = visualizer.visualize(img)
        mpimg.imsave(output_image_location + "/final_" + file, final)
