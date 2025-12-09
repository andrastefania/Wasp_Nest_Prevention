import cv2
import time
import sys
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.detection.background_subtractor import BackgroundSubtractor
from src.detection.motion_detector import MotionDetector
from src.detection.frequency_tracker import FrequencyTracker

class MainPipeline:

    def __init__(self, video_path=None):
        # 1. Background Subtractor
        self.bg = BackgroundSubtractor(
            history=500,
            var_threshold=16,
            detect_shadows=False
        )

        # 2. Motion Detector
        self.motion = MotionDetector(min_contour_area=20)

        # 3. Input Source
        if video_path is not None:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(0)

        self.freq = FrequencyTracker(
            frame_width=800,
            frame_height=450,
            cell_w=20,
            cell_h=20,
            decay_rate=0.001,
            hotspot_threshold=50   # threshold for nest detection
        )

        # 5. Optional log of all detections (if you want offline analysis)
        self.detection_log = []
        self.frame_idx = 0

        time.sleep(0.5)

    def run(self, mode="prevention"):
        """
        mode: 
          - "debug": Arată frame + mască + curățare + heatmap
          - "prevention": Overlay + scor activitate
        """
        print(f"Starting Wasp System in '{mode}' mode...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or cannot open camera.")
                break

            # --- PRE-PROCESARE ---
            frame = cv2.resize(frame, (800, 450))
            frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)

            # --- DETECTARE ---
            fg_mask = self.bg.apply(frame_blur)
            boxes, cleaned_mask = self.motion.detect(fg_mask, frame)
            
            for (x, y, w, h) in boxes:
                cx = int(x + w/2)
                cy = int(y + h/2)
                self.detection_log.append((self.frame_idx, cx, cy))

            self.frame_idx += 1
            
            self.freq.update(boxes)
            
            
            if mode == "debug":

                # Show original + bounding boxes
                debug_frame = frame.copy()
                for (x, y, w, h) in boxes:
                    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                freq_visual = self.freq.get_visual_map()

                cv2.imshow("1. Original Frame + Boxes", debug_frame)
                cv2.imshow("2. Raw Foreground Mask", fg_mask)
                cv2.imshow("3. Cleaned Mask", cleaned_mask)
                cv2.imshow("4. Frequency Hotspot Map", freq_visual)

            # -----------------------------------------------------------------
            # PREVENTION MODE (actual system)
            # -----------------------------------------------------------------
            elif mode == "prevention":

                freq_visual = self.freq.get_visual_map()
                overlay = cv2.addWeighted(frame, 0.7, freq_visual, 0.3, 0)

                # Draw bounding boxes on overlay
                for (x, y, w, h) in boxes:
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # --- Status Logic Based on Hotspots ---
                hotspot = self.freq.get_hotspot_value()

                nest = self.freq.is_nest_detected()

                # defaults
                status_text = "SAFE"
                text_color = (0, 255, 0)

                if nest:
                    status_text = "DANGER: NEST DETECTED"
                    text_color = (0, 0, 255)
                elif hotspot >= 20:
                    status_text = "WARNING: HIGH ACTIVITY"
                    text_color = (0, 165, 255)

                # Draw text
                cv2.putText(overlay, f"Hotspot Level: {hotspot:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.putText(overlay, f"Status: {status_text}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                cv2.imshow("Wasp Prevention System", overlay)

            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    pipeline = MainPipeline(video_path="data/raw/Gemini_Record3_longer.mp4")

    # ALEGE MODUL:
    #pipeline.run(mode="debug")
    pipeline.run(mode="prevention")
