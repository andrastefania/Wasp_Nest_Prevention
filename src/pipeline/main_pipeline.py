import cv2
import time
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.detection.background_subtractor import BackgroundSubtractor
from src.detection.motion_detector import MotionDetector
from src.detection.heatmap_tracker import HeatmapTracker


class MainPipeline:

    def __init__(self, video_path=None):

        # Background subtractor
        self.bg = BackgroundSubtractor(
            history=500,
            var_threshold=16,
            detect_shadows=False
        )

        # Motion detector
        self.motion = MotionDetector(
            min_contour_area=20
        )

        # Input source (video or webcam)
        if video_path is not None:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(0)

        # Heatmap (same size as resized frames)
        self.heatmap = HeatmapTracker(width=800, height=450)

        time.sleep(0.5)

    def run(self):
        while True:

            ret, frame = self.cap.read()
            if not ret:
                print("End of video or cannot open camera.")
                break

            # Standardize frame size
            frame = cv2.resize(frame, (800, 450))

            # Blur for smoother background subtraction
            frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)

            # 1. Background subtraction
            fg_mask = self.bg.apply(frame_blur)

            # 2. Motion detection
            boxes, cleaned_mask = self.motion.detect(fg_mask, frame)

            # 3. Draw bounding boxes
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

            # 4. Heatmap update
            self.heatmap.update(boxes)

            # 5. Get visual heatmap
            heat_img = self.heatmap.get_heatmap_visual()

            # 6. Show windows
            cv2.imshow("Frame", frame)
            cv2.imshow("Foreground Mask", fg_mask)
            cv2.imshow("Cleaned Mask", cleaned_mask)
            cv2.imshow("Heatmap", heat_img)

            # Exit with ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pipeline = MainPipeline(video_path="data/raw/Gemini_Record4.mp4")
    pipeline.run()
