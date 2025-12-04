import cv2
import time
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.detection.background_subtractor import BackgroundSubtractor
from src.detection.motion_detector import MotionDetector


class MainPipeline:
    """
    This class represents the FULL WASP DETECTION PIPELINE.

    It connects:
        - camera or input video
        - background subtractor (MOG2)
        - motion detector (contours + bounding boxes)
        - future modules (heatmap, surface change, classifier)

    The idea:
        For each frame:
            1. Read frame
            2. Compute foreground mask using MOG2
            3. Detect moving objects (insects)
            4. (Future) Update heatmap
            5. (Future) Surface change detection
            6. (Future) Save crops, send alert
            7. Display results for debugging
    """

    def __init__(self, video_path=None):
        """
        Initialize the entire pipeline.

        If video_path is provided:
            Load video from file (for software-only development phase)

        If video_path is None:
            Read from webcam or IP camera (future hardware integration)
        """

        # 1. Initialize the background subtractor
        self.bg = BackgroundSubtractor(
            history=500,
            var_threshold=16,
            detect_shadows=False
        )

        # 2. Initialize the motion detector
        self.motion = MotionDetector(
            min_contour_area=20
        )

        # 3. Decide input source (file or webcam)
        if video_path is not None:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(0)  # webcam

        # Small delay to stabilize capture
        time.sleep(0.5)

    def run(self):
        """
        Main loop of the system.
        Runs frame-by-frame until user closes the window.

        For each frame:
            - Capture frame
            - Resize for consistency
            - Apply background subtraction
            - Detect motion (bounding boxes)
            - (Future) Feed boxes into heatmap
            - (Future) Surface change detection
            - Draw results
            - Show windows
        """

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or cannot open camera.")
                break

            # Resize frame for consistent processing
            frame = cv2.resize(frame, (800, 450))

            frame_blur = cv2.GaussianBlur(frame, (7,7), 0)
            # 1. Background subtraction -> foreground mask
            fg_mask = self.bg.apply(frame_blur)

            # 2. Motion detection -> bounding boxes
            boxes, cleaned_mask = self.motion.detect(fg_mask, frame)


            # 3. Draw bounding boxes on frame
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

            # 4. Display debug windows
            cv2.imshow("Frame", frame)
            cv2.imshow("Foreground Mask", fg_mask)
            cv2.imshow("Cleaned Mask", cleaned_mask)

            # ESC key to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    pipeline = MainPipeline(video_path="data/raw/Gemini_Record3_longer.mp4")
    pipeline.run()