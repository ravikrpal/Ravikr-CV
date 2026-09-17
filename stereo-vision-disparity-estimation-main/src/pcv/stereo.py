import cv2
import numpy as np


def stereo_bm_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    num_disparities: int = 64,
    block_size: int = 15,
) -> np.ndarray:
    """
    Compute disparity using basic block matching (StereoBM).
    """
    # Ensure valid OpenCV parameters
    num_disparities = max(16, (num_disparities // 16) * 16)
    block_size = block_size if block_size % 2 == 1 else block_size + 1

    stereo = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size,
    )

    # Convert from fixed-point (scaled by 16) to float
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity


def stereo_sgbm_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    num_disparities: int = 96,
    block_size: int = 5,
) -> np.ndarray:
    """
    Compute disparity using Semi-Global Block Matching (SGBM).
    """
    # Ensure valid OpenCV parameters
    num_disparities = max(16, (num_disparities // 16) * 16)
    block_size = block_size if block_size % 2 == 1 else block_size + 1

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,

        # Smoothness penalties
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,

        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity


def stereo_ncc_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_disparity: int = 48,
    window_size: int = 9,
    scale: float = 0.5,
) -> np.ndarray:
    """
    Compute disparity using local Normalized Cross-Correlation (NCC).
    Used as a simple baseline for comparison.
    """
    # Ensure valid window size
    if window_size % 2 == 0:
        window_size += 1

    if scale <= 0 or scale > 1:
        raise ValueError("scale must be in (0, 1].")

    original_h, original_w = left_gray.shape

    # Downscale for speed (NCC is expensive)
    if scale != 1.0:
        left_small = cv2.resize(left_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        left_small = left_gray.copy()
        right_small = right_gray.copy()

    left = left_small.astype(np.float32)
    right = right_small.astype(np.float32)

    h, w = left.shape
    r = window_size // 2
    max_disp_scaled = max(1, int(round(max_disparity * scale)))

    disparity = np.full((h, w), -1, dtype=np.float32)

    # Match along horizontal scanlines
    for y in range(r, h - r):
        row_top = y - r
        row_bottom = y + r + 1

        for x in range(r + max_disp_scaled, w - r):
            template = left[row_top:row_bottom, x - r:x + r + 1]

            # Search region in right image
            x_min = x - max_disp_scaled - r
            x_max = x + r + 1
            if x_min < 0:
                continue

            search_strip = right[row_top:row_bottom, x_min:x_max]
            if search_strip.shape[1] < template.shape[1]:
                continue

            # NCC matching
            result = cv2.matchTemplate(search_strip, template, cv2.TM_CCORR_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)

            best_x = x_min + (max_loc[0] + r)
            disparity[y, x] = float(x - best_x)

    # Resize back to original resolution
    if scale != 1.0:
        disparity = cv2.resize(disparity, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        valid = disparity >= 0
        disparity[valid] /= scale

    return disparity