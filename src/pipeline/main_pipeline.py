# import cv2
# import time
# import sys
# import os
# import numpy as np

# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(ROOT_DIR)

# from src.detection.background_subtractor import BackgroundSubtractor
# from src.detection.motion_detector import MotionDetector
# from src.detection.heatmap_tracker import HeatmapTracker


# class MainPipeline:

#     def __init__(self, video_path=None):
#         # 1. Background Subtractor
#         self.bg = BackgroundSubtractor(
#             history=500,
#             var_threshold=16,
#             detect_shadows=False
#         )

#         # 2. Motion Detector
#         self.motion = MotionDetector(
#             min_contour_area=20
#         )

#         # 3. Input Source
#         if video_path is not None:
#             self.cap = cv2.VideoCapture(video_path)
#         else:
#             self.cap = cv2.VideoCapture(0)

#         # 4. Heatmap Tracker
#         # Ensure these dimensions match the resize in run()
#         self.heatmap = HeatmapTracker(width=800, height=450)
        
#         time.sleep(0.5)

#     def run(self, mode="prevention"):
#         """
#         mode: 
#           - "debug": Shows 4 separate windows (Frame, Mask, Cleaned, Heatmap)
#           - "prevention": Shows 1 combined window with Overlay + Stats
#         """
#         print(f"Starting Wasp System in '{mode}' mode...")

#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("End of video or cannot open camera.")
#                 break

#             # --- PRE-PROCESSING ---
#             frame = cv2.resize(frame, (800, 450))
#             frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)

#             # --- DETECTION ---
#             fg_mask = self.bg.apply(frame_blur)
#             boxes, cleaned_mask = self.motion.detect(fg_mask, frame)

#             # --- TRACKING ---
#             self.heatmap.update(boxes)
#             heat_img = self.heatmap.get_heatmap_visual()

#             # =========================================================
#             # OPTION 1: DEBUG MODE (4 Windows)
#             # =========================================================
#             if mode == "debug":
#                 # Draw boxes on the original frame
#                 debug_frame = frame.copy()
#                 for (x, y, w, h) in boxes:
#                     cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                 cv2.imshow("1. Original Frame", debug_frame)
#                 cv2.imshow("2. Raw Foreground", fg_mask)
#                 cv2.imshow("3. Cleaned Mask", cleaned_mask)
#                 cv2.imshow("4. Heatmap (Accumulated)", heat_img)

#             # =========================================================
#             # OPTION 2: PREVENTION MODE (All-in-One Overlay)
#             # =========================================================
#             elif mode == "prevention":
#                 # Create the overlay (70% Real World + 30% Heatmap)
#                 overlay = cv2.addWeighted(frame, 0.7, heat_img, 0.3, 0)

#                 # Draw boxes ON TOP of the overlay
#                 for (x, y, w, h) in boxes:
#                     cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                 # --- STATISTICS ---
#                 hot_pixels = np.sum(self.heatmap.map > 10)  # adjust threshold as needed
#                 activity_score = hot_pixels


#                 text_color = (0, 255, 0) # Green
#                 status_text = "SAFE"
                
#                 if activity_score > 1000:
#                     text_color = (0, 0, 255) # Red
#                     status_text = "DANGER: NEST DETECTED"

#                 cv2.putText(overlay, f"Activity Score: {activity_score}", (10, 30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
#                 cv2.putText(overlay, f"Status: {status_text}", (10, 60), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                
#                 cv2.imshow("Wasp Prevention System", overlay)

#             # Exit with ESC
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     pipeline = MainPipeline(video_path="data/raw/Gemini_Record3_longer.mp4")
    
#     # CHANGE THIS TO SWITCH MODES:
#     pipeline.run(mode="debug")      # Use this to tune masks/detection
#     #pipeline.run(mode="prevention") # Use this for the final presentation

import cv2
import time
import sys
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.detection.background_subtractor import BackgroundSubtractor
from src.detection.motion_detector import MotionDetector
from src.detection.heatmap_tracker import HeatmapTracker


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

        # 4. Heatmap Tracker (doar vizual)
        self.heatmap = HeatmapTracker(width=800, height=450)

        # 5. Activity scoring (bazat pe detecții, nu heatmap!)
        self.activity_history = []
        self.max_history = 1800  # ~1 minut la 30 FPS

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

            # --- HEATMAP (doar vizual) ---
            self.heatmap.update(boxes)
            heat_img = self.heatmap.get_heatmap_visual()

            # ==================================================================
            # MODE 1: DEBUG
            # ==================================================================
            if mode == "debug":
                debug_frame = frame.copy()
                for (x, y, w, h) in boxes:
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow("1. Original Frame", debug_frame)
                cv2.imshow("2. Raw Foreground", fg_mask)
                cv2.imshow("3. Cleaned Mask", cleaned_mask)
                cv2.imshow("4. Heatmap (Accumulated)", heat_img)

            # ==================================================================
            # MODE 2: PREVENTION SYSTEM (overlay + activitate)
            # ==================================================================
            elif mode == "prevention":

                # Creăm overlay-ul (70% real + 30% heatmap)
                overlay = cv2.addWeighted(frame, 0.7, heat_img, 0.3, 0)

                # Desenăm bounding box-urile
                for (x, y, w, h) in boxes:
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # ----------- ACTIVITY SCORE (BAZAT PE DETECȚII) -----------
                num_detectii = len(boxes)
                self.activity_history.append(num_detectii)

                # Păstrăm doar 1 minut de activitate (~1800 frame-uri)
                if len(self.activity_history) > self.max_history:
                    self.activity_history.pop(0)

                # Scor = total detecții în ultimul minut
                activity_score = sum(self.activity_history)

                # ----------- STATUS SISTEM -----------
                text_color = (0, 255, 0)  # Verde
                status_text = "SAFE"

                # Praguri realiste:
                # < 700 = activitate mică
                # 700–1500 = activitate moderată
                # > 1500 = cuib cert
                if activity_score > 1500:
                    text_color = (0, 0, 255)
                    status_text = "DANGER: NEST DETECTED"

                # Afișăm textul pe overlay
                cv2.putText(overlay, f"Activity Score (1 min): {activity_score}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.putText(overlay, f"Status: {status_text}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                cv2.imshow("Wasp Prevention System", overlay)

            # Taste ESC pentru ieșire
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pipeline = MainPipeline(video_path="data/raw/Gemini_Record3_longer.mp4")

    # ALEGE MODUL:
    #pipeline.run(mode="debug")
    pipeline.run(mode="prevention")
