import numpy as np
import cv2

class RoadVisualizer():

    def __init__(self, color, Minv):
        self.left_line = None
        self.right_line = None
        self.color = color
        self.Minv = Minv

        return

    def set_lane_lines(self, left, right):
        self.left_line = left
        self.right_line = right

        return

    def visualize(self, img):

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

        return result
