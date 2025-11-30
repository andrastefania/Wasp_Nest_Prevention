import cv2 
"""impotrs OpenCV library for computer vision tasks."""

class BackgroundSubtractor:
    """
    This class initializes and manages a background subtraction model.
    It is responsible for separating the STATIC part of the scene
    (the wall / corner where the nest might appear)
    from the MOVING objects (insects).

    Why we use background subtraction:
    - To detect motion in a video stream.
    - To ignore slow changes like daylight or shadows.
    - To isolate small, fast-moving objects such as wasps.

    We use OpenCV's MOG2 algorithm because:
    - It adapts over time (learning background changes).
    - It handles shadows well.
    - It is stable outdoors.
    """

    def __init__(self,
                 history=500,
                 var_threshold=16,
                 detect_shadows=True):
        """
        Constructor: initializes the MOG2 model.

        Parameters:
        - history (int): how many previous frames the model remembers.
                         Higher = smoother background, slower to adapt.
                         Lower = faster reaction, but more noise.
        - var_threshold (int): sensitivity for detecting movement.
                               Lower = detects smaller changes.
                               Higher = ignores small motion.
        - detect_shadows (bool): whether to mark shadows as separate regions.
                                 In outdoor scenes this helps reduce false positives.
        """

        # Create the MOG2 background subtractor
        self.model = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        """Internally: shadow pixels are marked as 127 (gray),moving objects are 255 (white),
           background is 0 (black)
        """

    def apply(self, frame):
        """
        Applies background subtraction to a single frame.

        Input:
        - frame: one image from the video stream

        Output:
        - fg_mask: foreground mask (black = background, white = motion)

        Explanation:
        The model compares the current frame with the learned background.
        Pixels that do NOT match the background model are marked as motion.
        """

        # Convert frame to grayscale for stability
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply the subtraction model
        fg_mask = self.model.apply(gray)

        return fg_mask
