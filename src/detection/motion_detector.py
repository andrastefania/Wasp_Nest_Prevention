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

    # def detect(self, fg_mask):
    #     """
    #     Cleans the mask, finds contours, filters them, and returns bounding boxes.
    #     """

    #     cleaned = self.clean_mask(fg_mask)

    #     contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     boxes = []

    #     for contour in contours:
    #         area = cv2.contourArea(contour)

    #         # Ignore tiny contours (noise)
    #         if area < self.min_contour_area:
    #             continue

    #         x, y, w, h = cv2.boundingRect(contour)
    #         boxes.append((x, y, w, h))

    #     return boxes, cleaned

    # def detect(self, fg_mask, frame=None):
    #     cleaned = self.clean_mask(fg_mask)
    #     contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     boxes = []

    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if area < self.min_contour_area:
    #             continue

    #         x, y, w, h = cv2.boundingRect(contour)
            
    #         # --- FILTER LOGIC ---
    #         is_wasp = True # Assume valid until proven otherwise

    #         # 1. Shape Check
    #         aspect_ratio = w / float(h)
    #         if aspect_ratio > 3.0 or aspect_ratio < 0.3:
    #             is_wasp = False

    #         # 2. Color/Contrast Check (The code we discussed)
    #         if is_wasp and frame is not None:
    #             # ... [Insert your Yellow/Contrast Logic here] ...
    #             roi = frame[y:y+h, x:x+w]
    #             hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #             # Define Wasp Yellow (Adjust these slightly if needed)
    #             lower_yellow = np.array([18, 50, 50])
    #             upper_yellow = np.array([35, 255, 255])
                
    #             mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
                
    #             # Count yellow pixels
    #             yellow_pixels = cv2.countNonZero(mask_yellow)
    #             total_pixels = w * h
    #             yellow_ratio = yellow_pixels / total_pixels

    #             if yellow_ratio < 0.05:
    #                 is_wasp = False

    #         # --- DECISION TIME ---
    #         if is_wasp:
    #             boxes.append((x, y, w, h))
    #         else:
    #             # OPTIONAL: Paint the rejected shadow BLACK on the mask
    #             # "cleaned" is the image we are editing
    #             # "-1" means fill the shape
    #             cv2.drawContours(cleaned, [contour], -1, 0, -1) 

    #     return boxes, cleaned

    def detect(self, fg_mask, frame=None):
        """
        Cleans the mask, finds contours, filters them, and returns bounding boxes.
        Uses shape, color, brightness and edge-strength heuristics to reject
        shadows/noise. Rejected contours are optionally painted black on the
        cleaned mask so they do not reappear in later processing.
        """

        cleaned = self.clean_mask(fg_mask)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # --- FILTER LOGIC ---
            is_wasp = True  # presupunem că e viespe până demonstrăm contrariul

            # 1. Shape check (elimină forme foarte lunguiețe sau foarte plate)
            aspect_ratio = w / float(h) if h > 0 else 0.0
            if aspect_ratio > 3.0 or aspect_ratio < 0.3:
                is_wasp = False

            # 2. Color + luminanță + textură (doar dacă încă pare viespe)
            if is_wasp and frame is not None:
                roi = frame[y:y+h, x:x+w]

                # If ROI is empty (shouldn't happen) skip
                if roi.size == 0:
                    is_wasp = False
                else:
                    # 2.a – Culoare galbenă (viespea are zone galbene)
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    lower_yellow = np.array([18,  80,  80])   # poți ajusta
                    upper_yellow = np.array([35, 255, 255])

                    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
                    yellow_pixels = cv2.countNonZero(mask_yellow)
                    total_pixels = max(w * h, 1)
                    yellow_ratio = yellow_pixels / float(total_pixels)

                    # 2.b – Luminozitate (umbrele sunt mai întunecate)
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray)

                    # 2.c – Textură (viespea are contur clar, umbra e blur)
                    edges = cv2.Canny(gray, 30, 60)
                    edge_strength = float(np.sum(edges))

                    # ---- Decizii ----
                    # dacă avem puțin galben SAU e foarte întunecat SAU contur foarte slab -> probabil umbră/zgomot
                    if yellow_ratio < 0.06 or mean_brightness < 40 or edge_strength < 200.0:
                        is_wasp = False

            # --- DECISION TIME ---
            if is_wasp:
                boxes.append((x, y, w, h))
            else:
                # colorăm conturul respins cu NEGRU în masca "cleaned" ca să dispară
                cv2.drawContours(cleaned, [contour], -1, 0, -1)

        return boxes, cleaned

