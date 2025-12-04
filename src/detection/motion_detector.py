import cv2
import numpy as np

class MotionDetector:
    """
    This class takes the foreground mask produced by the background subtractor
    and extracts MEANINGFUL moving objects (insects).

    Steps performed:
        1. Threshold (remove shadows, force 0/255)
        2. Morphological OPEN (remove noise)
        3. Morphological CLOSE (merge broken blobs)
        4. Contour detection + filtering
    """

    def __init__(self, min_contour_area=20):
        """
        Parameters:
        - min_contour_area: minimum pixel area to consider a real object.
        """
        self.min_contour_area = min_contour_area

    def clean_mask(self, fg_mask):
        """
        Removes noise from the raw foreground mask using:
        - Threshold
        - Morphological OPEN (erode → dilate)
        - Morphological CLOSE (dilate → erode)
        """

        # 1. Threshold: convert everything to clean 0/255
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 2. OPEN: remove small noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # 3. CLOSE: merge small fragmented blobs
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        return thresh

    def detect(self, fg_mask):
        """
        Cleans the mask, finds contours, filters them, and returns bounding boxes.
        """

        cleaned = self.clean_mask(fg_mask)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Ignore tiny contours (noise)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

        return boxes, cleaned
