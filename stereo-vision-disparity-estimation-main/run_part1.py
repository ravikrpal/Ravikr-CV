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

from pcv.io_utils import read_image, ensure_dir
from pcv.stereo import stereo_sgbm_disparity, stereo_ncc_disparity
from pcv.visualization import (
    save_disparity_figure,
    save_matches_figure,
    save_epipolar_figure,
)
from pcv.geometry import (
    detect_and_match_features,
    estimate_fundamental_matrix,
    compute_epilines,
    compute_epipole,
)


def main():
    root = Path(__file__).resolve().parent
    outputs = ensure_dir(root / "outputs" / "part1")

    left_path = root / "data" / "raw" / "left.png"
    right_path = root / "data" / "raw" / "right.png"

    print("\n--- DEBUG PATH CHECK ---")
    print("Project root:", root)
    print("Left path:", left_path)
    print("Right path:", right_path)
    print("Left exists:", left_path.exists())
    print("Right exists:", right_path.exists())
    print("------------------------\n")

    # Load images
    left_bgr = read_image(left_path, grayscale=False)
    right_bgr = read_image(right_path, grayscale=False)
    left_gray = read_image(left_path, grayscale=True)
    right_gray = read_image(right_path, grayscale=True)

    # 1) Disparity with SGBM
    disparity_sgbm = stereo_sgbm_disparity(
        left_gray,
        right_gray,
        num_disparities=96,
        block_size=5,
    )
    save_disparity_figure(
        disparity_sgbm,
        outputs / "disparity_sgbm.png",
        title="Part 1 disparity map (SGBM)",
    )

    # 1b) Disparity with NCC
    disparity_ncc = stereo_ncc_disparity(
        left_gray,
        right_gray,
        max_disparity=64,
        window_size=9,
    )
    save_disparity_figure(
        disparity_ncc,
        outputs / "disparity_ncc.png",
        title="Part 1 disparity map (NCC)",
    )

    # 2) Feature matching
    kpts1, kpts2, matches, pts1, pts2 = detect_and_match_features(
        left_gray,
        right_gray,
    )
    save_matches_figure(
        left_bgr,
        kpts1,
        right_bgr,
        kpts2,
        matches,
        outputs / "feature_matches.png",
    )

    # 3) Fundamental matrix
    F, inlier_mask = estimate_fundamental_matrix(pts1, pts2)
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    # 4) Epipolar lines
    n_show = min(12, len(pts1_in))
    pts1_show = pts1_in[:n_show]
    pts2_show = pts2_in[:n_show]

    lines_in_left = compute_epilines(
        pts2_show,
        which_image=2,
        F=F,
    )

    save_epipolar_figure(
        left_bgr,
        right_bgr,
        lines_in_left,
        pts1_show,
        pts2_show,
        outputs / "epipolar_lines.png",
    )

    # 5) Epipoles
    left_epipole = compute_epipole(F, left=True)
    right_epipole = compute_epipole(F, left=False)

    # 6) Summary
    summary = [
        "Part 1 summary",
        "===============",
        f"Total raw matches: {len(matches)}",
        f"RANSAC inliers: {len(pts1_in)}",
        "",
        "Estimated fundamental matrix F:",
        np.array2string(F, precision=5, suppress_small=True),
        "",
        "Left epipole (homogeneous, normalized):",
        np.array2string(left_epipole, precision=5, suppress_small=True),
        "",
        "Right epipole (homogeneous, normalized):",
        np.array2string(right_epipole, precision=5, suppress_small=True),
        "",
        "Stereo comparison:",
        "- SGBM: semi-global method with smoothness constraints, expected to produce cleaner disparity.",
        "- NCC: local correlation-based method, expected to be noisier and less stable in weakly textured regions.",
        "",
        "Note:",
        "This gives the relative epipolar geometry between the views.",
        "A metric rotation/translation pair would require camera intrinsics (calibration).",
    ]
    (outputs / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\nSaved Part 1 outputs to:")
    print(f"  {outputs}")
    for name in [
        "disparity_sgbm.png",
        "disparity_ncc.png",
        "feature_matches.png",
        "epipolar_lines.png",
        "summary.txt",
    ]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()