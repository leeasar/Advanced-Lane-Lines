import numpy as np
import cv2

""" This file includes a set of thresholding functions.
"""

def hls_select(img, channel, thresh=(0, 255)):
    """ Creates a binary map of the S (saturation) channel of the input image.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,channel]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def rgb_select(img, channel, thresh=(0, 255)):
    """ Creates a binary map of the selected R, G, or B channel of the input image.
    """
    color_channel = img[:,:,channel]
    binary_output = np.zeros_like(color_channel)
    binary_output[(color_channel > thresh[0]) & (color_channel <= thresh[1])] = 1
    return binary_output

def sobel_thresh(img, direction='x', thresh=(0, 255)):
    """ Creates a horizontal or vertical sobel threshold binary map
    from the input image.
    """
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
    """ Creates a sobel magnitude threshold binary map from the input image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    scaled = np.uint8(255*mag/np.max(mag))
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    return binary_output

def sobel_dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ Creates a directional sobel threshold binary map from tne input image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    arc = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(arc)
    binary_output[(arc >= thresh[0]) & (arc <= thresh[1])] = 1
    return binary_output

def canny_thresh(img, thresh=(0,255)):
    """ Creates a binary map that shows the Canny thresholds of the input image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, thresh[0], thresh[1])
    canny = np.uint8(255*canny)
    return canny
