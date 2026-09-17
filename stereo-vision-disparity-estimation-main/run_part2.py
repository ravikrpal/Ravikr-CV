import sys
import os

# ---------------- IMPORT PATH ----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# ---------------- IMPORTS ----------------
from pathlib import Path
import numpy as np
import cv2

from pcv.io_utils import read_image, ensure_dir
from pcv.stereo import stereo_sgbm_disparity, stereo_ncc_disparity
from pcv.geometry import (
    detect_and_match_features,
    estimate_fundamental_matrix,
    compute_epilines,
)
from pcv.rectification import uncalibrated_rectify
from pcv.visualization import (
    save_disparity_figure,
    save_matches_figure,
    save_epipolar_figure,
)


def draw_horizontal_epilines(img_bgr, pts, out_path):
    """Draw horizontal lines through rectified corresponding points."""
    img = img_bgr.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    pts = np.asarray(pts)
    rng = np.random.default_rng(0)

    for p in pts:
        x, y = p
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        x = int(round(x))
        y = int(round(y))
        cv2.line(img, (0, y), (img.shape[1] - 1, y), color, 1)
        cv2.circle(img, (x, y), 5, color, -1)

    cv2.imwrite(str(out_path), img)


def warp_points(H, pts):
    """Apply homography H to Nx2 points."""
    pts = np.asarray(pts, dtype=np.float32)
    pts_h = cv2.convertPointsToHomogeneous(pts).reshape(-1, 3).T
    warped = H @ pts_h
    warped /= warped[2:3, :]
    return warped[:2, :].T


def main():
    root = Path(__file__).resolve().parent
    outputs = ensure_dir(root / "outputs" / "part2")

    left_path = root / "data" / "captured" / "captured_left.png"
    right_path = root / "data" / "captured" / "captured_right.png"

    print("\n--- DEBUG PATH CHECK ---")
    print("Project root:", root)
    print("Left path:", left_path)
    print("Right path:", right_path)
    print("Left exists:", left_path.exists())
    print("Right exists:", right_path.exists())
    print("------------------------\n")

    # ---------------- LOAD IMAGES ----------------
    left_bgr = read_image(left_path, grayscale=False)
    right_bgr = read_image(right_path, grayscale=False)
    left_gray = read_image(left_path, grayscale=True)
    right_gray = read_image(right_path, grayscale=True)

    # ---------------- 1) DISPARITY BEFORE RECTIFICATION ----------------
    disparity_before_sgbm = stereo_sgbm_disparity(
        left_gray,
        right_gray,
        num_disparities=96,
        block_size=5,
    )
    save_disparity_figure(
        disparity_before_sgbm,
        outputs / "disparity_before_rectification_sgbm.png",
        title="Part 2 disparity before rectification (SGBM)",
    )

    disparity_before_ncc = stereo_ncc_disparity(
        left_gray,
        right_gray,
        max_disparity=48,
        window_size=9,
        scale=0.5,
    )
    save_disparity_figure(
        disparity_before_ncc,
        outputs / "disparity_before_rectification_ncc.png",
        title="Part 2 disparity before rectification (NCC)",
    )

    # ---------------- 2) FEATURE MATCHING ----------------
    kpts1, kpts2, matches, pts1, pts2 = detect_and_match_features(left_gray, right_gray)
    save_matches_figure(
        left_bgr,
        kpts1,
        right_bgr,
        kpts2,
        matches,
        outputs / "feature_matches_part2.png",
    )

    # ---------------- 3) FUNDAMENTAL MATRIX ----------------
    F, inlier_mask = estimate_fundamental_matrix(pts1, pts2)
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    if len(pts1_in) < 8:
        raise RuntimeError("Not enough inlier points for reliable rectification.")

    # ---------------- 4) EPIPOLAR LINES BEFORE RECTIFICATION ----------------
    n_show = min(12, len(pts1_in))
    pts1_show = pts1_in[:n_show]
    pts2_show = pts2_in[:n_show]

    lines_in_left = compute_epilines(pts2_show, which_image=2, F=F)
    save_epipolar_figure(
        left_bgr,
        right_bgr,
        lines_in_left,
        pts1_show,
        pts2_show,
        outputs / "epipolar_before_rectification.png",
    )

    # ---------------- 5) UNCALIBRATED RECTIFICATION ----------------
    rect_left_bgr, rect_right_bgr, H1, H2 = uncalibrated_rectify(
        left_bgr,
        right_bgr,
        pts1_in,
        pts2_in,
        F,
    )

    cv2.imwrite(str(outputs / "rectified_left.png"), rect_left_bgr)
    cv2.imwrite(str(outputs / "rectified_right.png"), rect_right_bgr)

    rect_left_gray = cv2.cvtColor(rect_left_bgr, cv2.COLOR_BGR2GRAY)
    rect_right_gray = cv2.cvtColor(rect_right_bgr, cv2.COLOR_BGR2GRAY)

    # ---------------- 6) EPIPOLAR LINES AFTER RECTIFICATION ----------------
    pts1_rect = warp_points(H1, pts1_show)
    pts2_rect = warp_points(H2, pts2_show)

    draw_horizontal_epilines(
        rect_left_bgr,
        pts1_rect,
        outputs / "epipolar_after_rectification_left.png",
    )
    draw_horizontal_epilines(
        rect_right_bgr,
        pts2_rect,
        outputs / "epipolar_after_rectification_right.png",
    )

    # ---------------- 7) DISPARITY AFTER RECTIFICATION ----------------
    disparity_after_sgbm = stereo_sgbm_disparity(
        rect_left_gray,
        rect_right_gray,
        num_disparities=96,
        block_size=5,
    )
    save_disparity_figure(
        disparity_after_sgbm,
        outputs / "disparity_after_rectification_sgbm.png",
        title="Part 2 disparity after rectification (SGBM)",
    )

    disparity_after_ncc = stereo_ncc_disparity(
        rect_left_gray,
        rect_right_gray,
        max_disparity=48,
        window_size=9,
        scale=0.5,
    )
    save_disparity_figure(
        disparity_after_ncc,
        outputs / "disparity_after_rectification_ncc.png",
        title="Part 2 disparity after rectification (NCC)",
    )

    # ---------------- 8) SUMMARY ----------------
    summary = [
        "Part 2 summary",
        "===============",
        f"Total raw matches: {len(matches)}",
        f"RANSAC inliers: {len(pts1_in)}",
        "",
        "Estimated fundamental matrix F:",
        np.array2string(F, precision=5, suppress_small=True),
        "",
        "Interpretation:",
        "- Disparity was computed on the original captured images.",
        "- Uncalibrated rectification was estimated from matched feature points and the fundamental matrix.",
        "- After rectification, epipolar lines should become approximately horizontal.",
        "- Stereo matching was repeated on the rectified images for comparison.",
        "",
        "Stereo comparison:",
        "- SGBM was used as the main method because it enforces smoothness and generally produces cleaner disparity maps.",
        "- NCC was implemented as a local baseline method for comparison.",
        "- Rectification is expected to improve both methods, but SGBM should remain more robust.",
        "",
        "Scene notes:",
        "- This scene contains large weakly textured wardrobe regions.",
        "- Better disparity is expected mainly near suitcase edges, handles, corners, and boundaries.",
        "- Flat wooden surfaces may still produce noisy or missing disparity values.",
    ]
    (outputs / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("Saved Part 2 outputs to:")
    print(f"  {outputs}")
    for name in [
        "feature_matches_part2.png",
        "epipolar_before_rectification.png",
        "rectified_left.png",
        "rectified_right.png",
        "epipolar_after_rectification_left.png",
        "epipolar_after_rectification_right.png",
        "disparity_before_rectification_sgbm.png",
        "disparity_before_rectification_ncc.png",
        "disparity_after_rectification_sgbm.png",
        "disparity_after_rectification_ncc.png",
        "summary.txt",
    ]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()