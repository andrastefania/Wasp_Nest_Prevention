import numpy as np
import cv2

class HeatmapTracker:
    def __init__(self, width, height, decay=0.995, max_heat=50.0):
        self.width = width
        self.height = height
        self.decay = decay
        self.max_heat = max_heat # A fixed cap prevents flickering

        # Floating heatmap for accumulation
        self.map = np.zeros((height, width), dtype=np.float32)

    def update(self, boxes):
        # 1. Apply decay (fades old trails)
        self.map *= self.decay

        # 2. Add activity
        # Instead of setting 1 pixel, we draw a soft circle (blob)
        for (x, y, w, h) in boxes:
            cx = int(x + w // 2)
            cy = int(y + h // 2)

            if 0 <= cx < self.width and 0 <= cy < self.height:
                # Add a Gaussian-like blob using a filled circle
                # We add distinct 'heat' rather than setting it
                # Radius depends on object size (optional) or fixed
                radius = 15 
                intensity = 2.0 
                
                # We use a temporary mask to add smoothness if desired, 
                # but adding a simple circle is much faster:
                cv2.circle(self.map, (cx, cy), radius, (intensity), -1)

    def get_heatmap_visual(self):
        """
        Converts heatmap into a visible image.
        """
        # 1. Clip values to a FIXED maximum. 
        # This prevents the map from constantly rescaling/flickering.
        # Any heat above self.max_heat is just "Max Red".
        clamped_map = np.clip(self.map, 0, self.max_heat)

        # 2. Normalize based on that FIXED range (0 to max_heat)
        # explicit scaling avoids the 'auto-scale' jitter
        heat_norm = (clamped_map / self.max_heat * 255).astype(np.uint8)

        # 3. Apply standard smoothing to blend the circles into blobs
        heat_norm = cv2.GaussianBlur(heat_norm, (15, 15), 0)

        # 4. Color map
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        return heat_color