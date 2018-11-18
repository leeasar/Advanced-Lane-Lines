import numpy as np

# Line is a class to track line detections over time
class Line():
    def __init__(self, side):
        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        #average x values of the fitted line over the last n iterations
        self.bestx = None

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels
        self.allx = None

        #y values for detected line pixels
        self.ally = None

        #is the line 'left' or 'right'
        self.side = side

        #last image we got
        self.last_img = False

        return

    def add_new_image(self, img):
        self.last_img = img
        self.find_new_coefs(img)
        return None

    def sanity_check_new_coefs(self):
        return None

    def find_new_coefs(self, img):
        # Find all pixel points that are roughly part of the line
        x_pix, y_pix = self.find_lane_points(img, nwindows=9, margin=100, minpix=50)
        # Fit a second order polynomial to the pixel points
        self.current_fit = np.polyfit(y_pix, x_pix, 2)

        # Sanity check for new coefficients ------------------------------- work on this

        # Decide about the next best coefficients ------------------------- work on this
        self.best_fit = self.current_fit
        print(self.current_fit)
        return None

    # Choose the number of sliding windows
    # Set the width of the windows +/- margin
    # Set minimum number of pixels found to recenter window
    # Set height of windows - based on nwindows above and image shape
    def find_lane_points(self, img, nwindows=9, margin=100, minpix=50):
        # Start by finding histogram peaks from the bottom half
        imshape = img.shape
        bottom_half = np.copy(img[int(img.shape[0]/2):imshape[0],:])
        histogram = np.sum(bottom_half, axis=0)
        midpoint = np.int(histogram.shape[0]//2)

        # Identify the starting positions
        base = 0
        if self.side == 'left':
            base = np.argmax(histogram[:midpoint])
        if self.side == 'right':
            base = np.argmax(histogram[midpoint:]) + midpoint

        if base == 0:
            print("No " + self.side + " lane was found") # ----------------------- so what happens then???
            return

        # Identify all lane pixel points using a sliding window
        window_height = np.int(img.shape[0]//nwindows)

        # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        current = base
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Decide window y boundaries
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            # Decide window x boundaries
            win_x_low = current - margin
            win_x_high = current + margin

            """
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2)
            """

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the list
            lane_inds.append(good_inds)

            # If we found > minpix pixels, recenter next window
            if len(good_inds) > minpix:
                current = np.int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)
        x_pix = nonzerox[lane_inds]
        y_pix = nonzeroy[lane_inds]

        return x_pix, y_pix


    def get_lane_pixels(self):
        ploty = np.linspace(0, self.last_img.shape[0]-1, self.last_img.shape[0])
        plotx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        return plotx, ploty
