import numpy as np

class Line():
    """ Line is a class to model a lane line and to track that lane line over time.
    """

    def __init__(self, side):
        """ Line class initializer.
        """
        self.recent_xfitted = [] # x values of historical fits of the lane
        self.best_fit = None # polynomial coefficients averaged over the last n iterations
        self.current_fit = [np.array([False])] # polynomial coefficients for the most recent fit
        self.radius_of_curvature = None # radius of curvature of the line in some units
        self.side = side # is the line 'left' or 'right'
        self.last_img = False # last image processed
        self.confidence = 0 # confidence value for the last line identification

        return

    def add_new_image(self, img):
        """ Extract lane information from images. The function assumes consequtive,
        preprocessed road images.

        Args:
            img (binary image): Preprocessed road image (undistorted, line thresholds, perspective warp)

        Returns:
            Nothing.
        """
        self.last_img = img

        # If the object is new or lane confidence from the previous image was low,
        # find lane pixel points using sliding window technique
        if self.confidence <= 0.1:
            x_pix, y_pix = self.find_lane_points_sliding_win(img, nwindows=9, margin=100, minpix=50)
            if len(x_pix) == 0:
                self.confidence = 0
                return None
        # If the lane is already previously known, find lane pixels using look ahead technique
        # If look ahead does not find pixels, revert to sliding window
        else:
            self.confidence -= 0.1
            x_pix, y_pix = self.find_lane_points_look_ahead(img, margin=125)
            if len(x_pix) == 0:
                x_pix, y_pix = self.find_lane_points_sliding_win(img, nwindows=9, margin=100, minpix=50)
                if len(x_pix) == 0:
                    self.confidence = 0
                    return None

        # Computing candidates for new fit and new curvature
        fit_candidate = np.polyfit(y_pix, x_pix, 2)
        y_eval = np.max(y_pix)
        curve_candidate = np.power((1 + (2*fit_candidate[0]*y_eval + fit_candidate[1])**2), 1.5)/np.absolute(2*fit_candidate[0])

        # When no previous values, proceed directly to saving the candidate values
        # because sanity-checking against history is not possible
        if self.confidence <= 0.1:
            self.current_fit = fit_candidate
            self.recent_xfitted.append(fit_candidate)
            self.best_fit = self.current_fit
            self.radius_of_curvature = np.copy(curve_candidate)
            self.confidence = 1.0

        # Sanity check -- if new polynom coefficients do not pass the sanity check, discard
        # Otherwise save the candidates as current values
        if self.sanity_check_new_coefs(fit_candidate):
            self.current_fit = fit_candidate
            self.recent_xfitted.append(fit_candidate)
            self.best_fit = self.current_fit
            self.radius_of_curvature = np.copy(curve_candidate)
        else:
            self.confidence = 0

        return None

    def sanity_check_new_coefs(self, fit_candidate):
        """ Doing a sanity check that the candidate fit coefficients are similar
        enough with the previous, i.e. possible.

        Args:
            fit_candidate (array): The candidate fit coefficients

        Returns:
            True or False for passing the sanity check.
        """
        if (np.absolute(fit_candidate[0] - self.current_fit[0]) < 25) and (np.absolute(fit_candidate[2] - self.current_fit[2]) < 70):
            return True
        else:
            return False

    # Choose the number of sliding windows
    # Set the width of the windows +/- margin
    # Set minimum number of pixels found to recenter window
    # Set height of windows - based on nwindows above and image shape
    def find_lane_points_sliding_win(self, img, nwindows=9, margin=100, minpix=50):
        """ Finds a list of pixel coordinates that form a lane line. Uses the sliding
        windows technique.

        Args:
            img (binary image): Input image
            nwindows (int): Number of vertical windows the image is split into
            margin (int): Defines the width of a window as margins around a center
            minpix (int): The minimum number of activated pixels that justifies moving the window horizontally

        Returns:
            x_pix (array): A list of x coordinates of activated pixels
            y_pix (array): A list of y coordinates of activated pixels
        """
        # Start by finding histogram peaks from the bottom half
        imshape = img.shape
        bottom_half = np.copy(img[int(img.shape[0]/2):imshape[0],:])
        histogram = np.sum(bottom_half, axis=0)
        midpoint = np.int(histogram.shape[0]//2)

        # Identify the starting position
        base = 0
        initial_margin = 100
        if self.side == 'left':
            base = np.argmax(histogram[initial_margin:midpoint-initial_margin]) + initial_margin
        if self.side == 'right':
            base = np.argmax(histogram[midpoint + initial_margin:]) + midpoint + initial_margin

        if base == 0:
            print("No " + self.side + " lane was found")
            return

        # Identify all lane pixel points using a sliding window
        window_height = np.int(img.shape[0] // nwindows)

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

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the list
            lane_inds.append(good_inds)

            # If we found > minpix pixels, recenter next window
            if len(good_inds) > minpix:
                current = np.int(np.mean(nonzerox[good_inds]))

        # Roll out the indices into a one-dimensional array
        # Based on the indices, create arrays of x and y lane pixel coordinates
        lane_inds = np.concatenate(lane_inds)
        x_pix = nonzerox[lane_inds]
        y_pix = nonzeroy[lane_inds]

        return x_pix, y_pix

    def find_lane_points_look_ahead(self, img, margin=100):
        """ Finds a list of pixel coordinates that form a lane line. Uses the
        look ahead technique.

        Args:
            img (binary image): Input image
            margin (int): Defines the width of the search area around the current lane line

        Returns:
            x_pix (array): A list of x coordinates of activated pixels
            y_pix (array): A list of y coordinates of activated pixels
        """
        # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lane_inds = []

        lane_inds = (
        (nonzerox >= self.current_fit[0]*nonzeroy**2 + self.current_fit[1]*nonzeroy + self.current_fit[2] - margin) &
        (nonzerox < self.current_fit[0]*nonzeroy**2 + self.current_fit[1]*nonzeroy + self.current_fit[2] + margin))

        x_pix = nonzerox[lane_inds]
        y_pix = nonzeroy[lane_inds]

        return x_pix, y_pix

    def get_lane_pixels(self):
        """ Returns the list of pixels forming the lane line as currently modeled.

        Returns:
            plotx (array): A list of x coordinates of the lane line
            ploty (array): A list of y coordinates of the lane line
        """
        ploty = np.linspace(0, self.last_img.shape[0]-1, self.last_img.shape[0])
        plotx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        return plotx, ploty

    def get_lane_curvature(self):
        """ Returns the radius of the lane curvature in meters.

        Returns:
            curverad (int): Returns the radius of the lane curvature in meters.
        """
        # Meters per pixel in y dimension
        ym_per_pix = 17/720
        # Define y-value where we want radius of curvature
        y_eval = 600
        # Calculation of R_curve (radius of curvature)
        curverad = np.power((1 + (2*self.current_fit[0]*y_eval*ym_per_pix + self.current_fit[1])**2), 1.5)/np.absolute(2*self.current_fit[0])

        return curverad

    def get_distance_from_center(self):
        """ Calculates the car's center's distance from the lane line meters

        Returns:
            from_center (int): The car's center's distance from the lane line in meters
        """
        # Meters per pixel in x dimension
        xm_per_pix = 3.7/850
        y_eval = self.last_img.shape[0]
        x = self.best_fit[0]*y_eval**2 + self.best_fit[1]*y_eval + self.best_fit[2]
        from_center = np.absolute(self.last_img.shape[1]/2 - x) * xm_per_pix
        return from_center
