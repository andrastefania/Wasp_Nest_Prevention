import cv2
import numpy as np

class MotionDetector:
    """
    This class takes the foreground mask produced by the background subtractor
    and extracts MEANINGFUL moving objects (insects).

    The raw mask is just pixels:
        - 0 = background
        - 127 = shadow
        - 255 = motion
        
    This module:
        1. Cleans noise (erosion + dilation)
        2. Finds contours (white pixel blobs)
        3. Filters out tiny noise contours
        4. Computes bounding boxes for each detected object

    The output is a list of bounding boxes:
        [(x, y, w, h), (x2, y2, w2, h2), ...]
    """

    def __init__(self,
                 min_contour_area=20,
                 erode_iterations=1,
                 dilate_iterations=2):
        """
        Constructor for MotionDetector.

        Parameters:
        - min_contour_area: minimum pixel area to consider something a real object.
          Tiny contours (1–10 px) are usually noise.

        - erode_iterations: how many times to erode (remove noise)

        - dilate_iterations: how many times to dilate (restore + slightly enlarge real objects)

        These values will be tuned later after testing on real videos.
        """

        self.min_contour_area = min_contour_area
        self.erode_iterations = erode_iterations
        self.dilate_iterations = dilate_iterations

    def clean_mask(self, fg_mask):
        """
        Takes the raw mask from MOG2 and removes noise.

        Why we do this:
        - Shadows create gray pixels (value 127)
        - Light flicker generates scattered white dots
        - Insects appear as solid shapes, not 1-pixel specks

        Steps performed:
        1. Threshold: convert everything to pure 0 or 255
        2. Erode: remove isolated noise dots
        3. Dilate: expand real moving objects slightly
        """

        # 1. Threshold the mask to get clean binary image
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 2. Remove small white noise (erode shrinks white areas)
        thresh = cv2.erode(thresh, None, iterations=self.erode_iterations)

        # 3. Expand remaining objects (dilate expands white areas)
        thresh = cv2.dilate(thresh, None, iterations=self.dilate_iterations)

        return thresh

    def detect(self, fg_mask):
        """
        Main function that:
        - cleans the mask
        - finds contours
        - filters them
        - returns bounding boxes

        Output:
        List of bounding boxes [(x, y, w, h), ...]
        """

        # Step 1: Clean the raw mask
        cleaned = self.clean_mask(fg_mask)

        # Step 2: Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        # Step 3: Loop through each contour
        for contour in contours:
            area = cv2.contourArea(contour)

            # Ignore tiny shapes (noise)
            if area < self.min_contour_area:
                continue

            # Compute bounding box
            x, y, w, h = cv2.boundingRect(contour)

            boxes.append((x, y, w, h))

        return boxes, cleaned
