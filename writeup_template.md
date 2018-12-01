## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric][1] Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here][2] is a template writeup for this project you can use as a guide and a starting point.  

This report in itself is the writeup deliverable.

### Camera Calibration

#### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `calibrate.py`.  I needed to establish object points and image points to use `cv2.calibrateCamera()` function to compute the camera calibration and distortion coefficients. I established the image points from the provided chessboard images by using `cv2.findChessboardCorners()` function. I established the object points by manually creating a 2D grid. The object points were identical for all calibration images.

Finally, I saved the calibration and distortion coefficients in a Pickle file for later use: `camera_cal/calibration_pickle.p`.

I tested the that the calibration and distortion coefficients were correct by running `cv2.undistort()` to a test image. Here are the original and the calibrated test image:

![alt text][image-1]![alt text][image-2]

### Pipeline for road images (single images)

Here is a starting point image for the pipeline:

![alt text][image-3]

#### 1. Provide an example of a distortion-corrected image.

I applied distortion correction by using `cv2.undistort()` function and the calibration and distortion coefficients saved in the calibration step:  

![alt text][image-4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Next, I used a combination of color and gradient thresholds to generate a binary image. I created a set of thresholding utility functions that are located in `thresholding_uti.py` file. I created function `thresholding()` on line 58 in `main.py` file and experimented with thresholding with RGB channels, HLS channels, horizontal, vertical and directional Sobel thresholds, and Canny thresholds. Eventually, I selected a combination of R channel, S channel and horizontal Sobel. This takes place on line 82:

```python
combined_binary[((red_bin == 1) & (hls_bin == 1)) | ((red_bin == 1) & (sobel_bin == 1))] = 1
```

Here's an example of my output for this step:

![alt text][image-5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Next, I do a perspective transform to create a “bird’s eye view”.  The code for this is located in `warp()` function in `main.py` file. The perspective warp is ultimately done by cv2.warpPerspective() function which needs source and destination points for warping. I chose the hardcode them in the following manner:

```python
src = np.float32([[588, 460], [697, 460], [1030, 710], [275, 710]])
dst = np.float32([[270, 0], [990, 0], [990, 710], [270, 710]])
```

I verified that my perspective transform was working as expected by applying it to a test image and ensuring that the distance between the left and right lane lines remains roughly constant throughout the image:

![alt text][image-6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I built to methods to identify lane line pixels in images: sliding window and look ahead. Both of these are functions of class `Line.py` (a separate file). Function `find_lane_points_sliding_win()` is located on line 95. Function `find_lane_points_look_ahead()` is located on line 167.

The sliding window technique identifies the lane locations at the bottom of the image through a histogram, and then a window by window keeps shifting the search area where the most pixels are as the search progresses towards the top of the image. The end result is a list of activated lane line pixels.

The look ahead technique establishes a search area around the lane line as it was identified in the previous image. It identifies all activated pixels in that zone and, similarly, the end result is a list of activated lane line pixels.

I then fitted these activated lane line pixels to a second order polynomial. This second order polynomial literally is my lane line detection.

I alternated between the two search functions based on the confidence level of the previous lane detection. The code of this is located in `add_new_image()`on line 20. When the confidence level for the previous lane detection is high, the look ahead technique is used. If the confidence is low or no line was found in the previous image, the function reverts to using the sliding window technique. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of radius of curvature takes place in the file `Line.py` in `get_lane_curvature()` function on line 206. The function uses the following formula to calculate the curvature. The curvature is calculated separately for the left and the right line. 

```python
curverad = np.power((1 + (2*self.current_fit[0]*y_eval*ym_per_pix + self.current_fit[1])**2), 1.5)/np.absolute(2*self.current_fit[0])
```

I use the values of `y_eval = 600` and `ym_per_pix = 17/720`, specifying what is the height where the curvature is calculated and what is the meters per pixel ratio vertically. Beyond that, the only thing that affects the curvature value is the lane line polynomial coefficients. 

The calculation of vehicle position with respect to center takes place in `get_distance_from_center()` function. This function calculates how far that particular line is from the image center in pixels, and then translates that into meters. The meters per pixel ratio that I used is `xm_per_pix = 3.7/850`.  These per single line measurements are then used later in `RoadVisualizer.py` to calculate the vehicle position with respect to center (line 64):

```python
from_center = np.absolute((self.left_line.get_distance_from_center() - self.right_line.get_distance_from_center())/2)

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Plotting the result back onto the road takes place in the `RoadVisualizer.py` file, in function `visualize()`.  Here is an example of my result on a test image:

![alt text][image-7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][3]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My biggest challenge was to establish such thresholding that would adequately capture the lane lines but would not capture irrelevant road noise that would falsely trigger lane line detection. I came to a conclusion that this is not possible with static thresholding parameters. Instead, the parameters would need to be dynamically altered based on road conditions.

For those images in which thresholding correctly highlighted the lane lines (and not too much extra noise) the lane detection worked extremely well. Because the lane detection was so robust, I ended up not implementing any smoothing by averaging for the lane line coefficients. Instead, each frame is processed individually. The handful of failed detections or where the detection did not pass a sanity check, the frame is simply ignored. Improving this logic for robust performance in challenging conditions would be another improvement area.


[1]:	https://review.udacity.com/#!/rubrics/571/view
[2]:	https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md
[3]:	./output_videos/project_video.mp4

[image-1]:	./camera_cal/calibration5.jpg "Image before calibrating"
[image-2]:	./camera_cal/test_calibration.jpg "Testing calibration"
[image-3]:	./test_images/test5.jpg "Original road image"
[image-4]:	./output_images/undist_test5.jpg "Undistortion applied"
[image-5]:	./output_images/thresh_test5.jpg "Thresholding applied"
[image-6]:	./output_images/warp_test5.jpg "Warping applied"
[image-7]:	./output_images/final_test5.jpg "Road info visualization applied"