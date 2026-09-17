import cv2
import numpy as np


def uncalibrated_rectify(img1, img2, pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray):
    h, w = img1.shape[:2]
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts1.astype(np.float32), pts2.astype(np.float32), F, imgSize=(w, h)
    )
    if not ok:
        raise RuntimeError("Rectification failed.")
    rect1 = cv2.warpPerspective(img1, H1, (w, h))
    rect2 = cv2.warpPerspective(img2, H2, (w, h))
    return rect1, rect2, H1, H2
