import numpy as np
import cv2

class FrequencyTracker:
    """
    Tracks spatial frequency of insect detections over time.
    Includes:
        - Cell grid accumulation
        - Leakage (decay)
        - Normalized hotspot map
        - Optional visual representation
    """

    def __init__(self, frame_width=800, frame_height=450, cell_w=20, cell_h=20,
                 decay_rate=0.001, hotspot_threshold=50):
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cell_w = cell_w
        self.cell_h = cell_h

        self.grid_cols = frame_width  // cell_w
        self.grid_rows = frame_height // cell_h

        # float32 because we apply decay
        self.freq_map = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        # decay configuration
        self.decay_rate = decay_rate
        self.hotspot_threshold = hotspot_threshold

    # ---------------------------------------------------------
    # UPDATE WITH NEW DETECTIONS
    # ---------------------------------------------------------
    def update(self, boxes):
        """
        Add new wasp detections and apply decay on each frame.
        boxes = [(x, y, w, h), ...]
        """

        # 1. DECAY (Leak)
        self.freq_map -= self.decay_rate
        np.clip(self.freq_map, 0, None, out=self.freq_map)

        # 2. ADD new detections to appropriate cells
        for (x, y, w, h) in boxes:
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            gx = cx // self.cell_w
            gy = cy // self.cell_h

            if 0 <= gx < self.grid_cols and 0 <= gy < self.grid_rows:
                self.freq_map[gy, gx] += 1

    # ---------------------------------------------------------
    # HOTSPOT ANALYSIS
    # ---------------------------------------------------------
    def get_hotspot_value(self):
        return float(np.max(self.freq_map))

    def is_nest_detected(self):
        return self.get_hotspot_value() >= self.hotspot_threshold

    # ---------------------------------------------------------
    # OPTIONAL VISUALIZATION FOR DEBUG
    # ---------------------------------------------------------
    def get_visual_map(self):
        """
        Returns an 800x450 color heatmap representing hotspots.
        Only for debug.
        """
        if self.freq_map.max() > 0:
            norm = np.clip(self.freq_map / self.hotspot_threshold, 0, 1)
        else:
            norm = self.freq_map

        norm = (norm * 255).astype(np.uint8)
        norm_resized = cv2.resize(norm, (self.frame_width, self.frame_height),
                                  interpolation=cv2.INTER_NEAREST)

        colored = cv2.applyColorMap(norm_resized, cv2.COLORMAP_JET)
        return colored
