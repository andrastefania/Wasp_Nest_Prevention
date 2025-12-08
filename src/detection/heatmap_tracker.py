import numpy as np
import cv2

# class HeatmapTracker:
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height
        
#         # --- TUNING THE "BUCKET" ---
#         self.max_heat = 100.0       # The "Red" limit
#         self.add_amount = 5.0       # Heat added per wasp (Water In)
        
#         # Heat removed per frame (The Leak)
#         # 0.05 means it takes 100 frames (~3 seconds) to lose 5 heat.
#         # If you want trails to last 1 minute: 5.0 / (60*30) = 0.0027
#         self.cooldown_rate = 0.01   
#         # ---------------------------

#         self.map = np.zeros((height, width), dtype=np.float32)

#     def update(self, boxes):
#         # 1. THE LEAK: Subtract cooldown from EVERYTHING
#         self.map -= self.cooldown_rate
        
#         # 2. Clamp to 0: Ensure we don't go into negative numbers (blacker than black)
#         self.map = np.maximum(self.map, 0)

#         # 3. WATER IN: Add heat for detections
#         for (x, y, w, h) in boxes:
#             cx = int(x + w // 2)
#             cy = int(y + h // 2)

#             if 0 <= cx < self.width and 0 <= cy < self.height:
#                 # Add heat
#                 cv2.circle(self.map, (cx, cy), 15, (self.add_amount), -1)

#     def get_heatmap_visual(self):
#         # Clip at max_heat so we don't get values like 5000
#         clamped_map = np.clip(self.map, 0, self.max_heat)

#         # Normalize 0 to max_heat
#         heat_norm = (clamped_map / self.max_heat * 255).astype(np.uint8)
        
#         heat_norm = cv2.GaussianBlur(heat_norm, (15, 15), 0)
#         return cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

# class HeatmapTracker:
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height

#         # --- NEW TUNING (Much more visible heatmap) ---
#         self.max_heat = 50.0          # Lower max -> colors look stronger
#         self.add_amount = 8.0         # More heat per detection
#         self.cooldown_rate = 0.002    # Trails last much longer (about 1 minute)
#         self.radius = 18              # Heat circle radius
#         # ------------------------------------------------

#         self.map = np.zeros((height, width), dtype=np.float32)

#     def update(self, boxes):
#         # Fade the heatmap slowly
#         self.map -= self.cooldown_rate
#         self.map = np.maximum(self.map, 0)

#         # Add heat at detection centers
#         for (x, y, w, h) in boxes:
#             cx = int(x + w // 2)
#             cy = int(y + h // 2)

#             if 0 <= cx < self.width and 0 <= cy < self.height:
#                 cv2.circle(self.map, (cx, cy), self.radius, (self.add_amount), -1)

#     def get_heatmap_visual(self):
#         # Clip values to max_heat
#         clamped = np.clip(self.map, 0, self.max_heat)

#         # Normalize to 0..255
#         normalized = (clamped / self.max_heat * 255).astype(np.uint8)

#         # Strong blur for smooth trails
#         normalized = cv2.GaussianBlur(normalized, (21, 21), 0)

#         # Color with JET colormap
#         heatmap_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

#         return heatmap_color

class HeatmapTracker:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.max_heat = 100
        self.add_amount = 10.0
        self.cooldown_rate = 0.002
        self.base_radius = 18

        self.map = np.zeros((height, width), dtype=np.float32)

    def update(self, boxes, dt=1.0):
        # Decay heat based on time delta
        self.map -= self.cooldown_rate * dt
        self.map = np.maximum(self.map, 0)

        for (x, y, w, h) in boxes:
            cx = int(x + w // 2)
            cy = int(y + h // 2)

            r = max(6, int(min(w, h) * 0.6))  # Scaled radius
            if 0 <= cx < self.width and 0 <= cy < self.height:
                cv2.circle(self.map, (cx, cy), r, self.add_amount, -1)

    def get_heatmap_visual(self):
        clamped = np.clip(self.map, 0, self.max_heat)
        norm = (clamped / self.max_heat * 255).astype(np.uint8)
        norm = cv2.GaussianBlur(norm, (21,21), 0)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)
