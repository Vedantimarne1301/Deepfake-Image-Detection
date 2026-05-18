"""
face_detect.py — Detect and crop the primary face from an image using OpenCV.
"""

import cv2
import numpy as np


# Haar cascade bundled with OpenCV — no extra download needed
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def detect_and_crop_face(img_bgr: np.ndarray,
                          target_size: int = 224,
                          padding: float = 0.2) -> tuple[np.ndarray | None, tuple | None]:
    """
    Detect the largest face in the image and return a cropped + resized patch.

    Args:
        img_bgr     : BGR image (H×W×3 uint8)
        target_size : output square size (pixels)
        padding     : fractional padding around the detected face box

    Returns:
        (face_crop, bbox) where face_crop is (target_size, target_size, 3) uint8
        and bbox is (x, y, w, h).  Returns (None, None) if no face is found.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None, None

    # Pick the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add padding
    ih, iw = img_bgr.shape[:2]
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)

    face_crop = img_bgr[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (target_size, target_size))

    return face_resized, (x1, y1, x2 - x1, y2 - y1)