import numpy as np
import cv2

class RoadVisualizer():
    """ RoadVisualizer is a class that does lane and road data visualization.
    """

    def __init__(self, color, Minv):
        self.left_line = None
        self.right_line = None
        self.color = color
        self.Minv = Minv

        return

    def set_lane_lines(self, left, right):
        """ Sets the left and right lane line for the visualizer. The line objects
        will include the data that gets visualized.

        Args:
            left (Line): The left lane line
            right (Line): The right lane line
        """
        self.left_line = left
        self.right_line = right

        return

    def visualize(self, img):
        """ Visualizes the lane area with a highlight color. Visualizes road data
        (curvature, distance from road center) as numbers.

        Args:
            img (RGB image): The input image, an undistorted RGB image

        Returns:
            The created visualization elements embedded into the input image.

        """
        # Create an image to draw the lines on
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Get x and y points for both lanes (y is actually identifcal)
        left_fitx, ploty = self.left_line.get_lane_pixels()
        right_fitx, ploty = self.right_line.get_lane_pixels()

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), self.color)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        # Get line curvatures
        left_curve = self.left_line.get_lane_curvature().astype(int).astype(str)
        right_curve = self.right_line.get_lane_curvature().astype(int).astype(str)
        from_center = np.absolute((self.left_line.get_distance_from_center() - self.right_line.get_distance_from_center())/2)
        from_center = round(from_center, 2).astype(str)

        # Add text to the image
        line1 = "Left curvature: " + left_curve + " m"
        line2 = "Right curvature: " + right_curve + " m"
        line3 = "Distance from center: " + from_center + " m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, line1, (20, 30), font, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(result, line2, (20, 70), font, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(result, line3, (20, 110), font, 1, (0,255,0), 2, cv2.LINE_AA)
        
        return result
