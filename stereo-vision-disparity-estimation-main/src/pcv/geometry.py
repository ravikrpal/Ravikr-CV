import cv2
import numpy as np


def detect_and_match_features(
    img1_gray: np.ndarray,
    img2_gray: np.ndarray,
    max_features: int = 2000,
):
    """
    Detect ORB features in both images and compute brute-force matches.
    Returns keypoints, matches, and corresponding point coordinates.
    """
    # Detect ORB keypoints and descriptors
    detector = cv2.ORB_create(nfeatures=max_features)
    kpts1, desc1 = detector.detectAndCompute(img1_gray, None)
    kpts2, desc2 = detector.detectAndCompute(img2_gray, None)

    if desc1 is None or desc2 is None:
        raise RuntimeError("Could not detect enough features for matching.")

    # Match descriptors using Hamming distance (suitable for ORB)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < 8:
        raise RuntimeError("Need at least 8 matches to estimate the fundamental matrix.")

    # Extract matched point coordinates
    pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches])

    return kpts1, kpts2, matches, pts1, pts2


def estimate_fundamental_matrix(points1: np.ndarray, points2: np.ndarray):
    """
    Estimate the fundamental matrix using RANSAC to reject outliers.
    """
    # Use RANSAC to obtain a robust estimate of F
    F, mask = cv2.findFundamentalMat(
        points1,
        points2,
        cv2.FM_RANSAC,
        1.0,   # reprojection threshold
        0.99,  # confidence
    )

    if F is None or F.shape != (3, 3):
        raise RuntimeError("Fundamental matrix estimation failed.")

    # Normalize F for numerical stability
    F = F / np.linalg.norm(F)

    return F, mask.ravel().astype(bool)


def compute_epilines(points: np.ndarray, which_image: int, F: np.ndarray):
    """
    Compute epipolar lines corresponding to input points.
    """
    lines = cv2.computeCorrespondEpilines(
        points.reshape(-1, 1, 2),
        which_image,
        F,
    )
    return lines.reshape(-1, 3)


def compute_epipole(F: np.ndarray, left: bool = False) -> np.ndarray:
    """
    Compute the epipole from the fundamental matrix.
    Returns right epipole by default, or left if left=True.
    """
    # Use SVD: epipole is the null space of F (or F^T)
    M = F.T if left else F
    _, _, vt = np.linalg.svd(M)

    e = vt[-1]

    # Normalize homogeneous coordinates
    e = e / e[-1]
    return e