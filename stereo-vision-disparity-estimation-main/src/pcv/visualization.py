from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np



def save_disparity_figure(disparity, out_path: str | Path, title: str = "Disparity map"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_less_equal(disparity, 0)
    plt.figure(figsize=(8, 5))
    plt.imshow(masked, cmap="gray")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def save_matches_figure(img1_bgr, kpts1, img2_bgr, kpts2, matches, out_path: str | Path, max_matches: int = 60):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = cv2.drawMatches(
        img1_bgr,
        kpts1,
        img2_bgr,
        kpts2,
        matches[:max_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out_path), vis)



def _draw_lines(img_a, img_b, lines, pts_a, pts_b):
    h, w = img_a.shape[:2]
    out_a = img_a.copy()
    out_b = img_b.copy()
    for line, pt_a, pt_b in zip(lines, pts_a, pts_b):
        color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
        a, b, c = line
        if abs(b) < 1e-8:
            continue
        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-(c + a * w) / b)
        cv2.line(out_a, (x0, y0), (x1, y1), color, 1)
        cv2.circle(out_a, tuple(np.round(pt_a).astype(int)), 5, color, -1)
        cv2.circle(out_b, tuple(np.round(pt_b).astype(int)), 5, color, -1)
    return out_a, out_b



def save_epipolar_figure(img1_bgr, img2_bgr, lines1, pts1, pts2, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    left, right = _draw_lines(img1_bgr, img2_bgr, lines1, pts1, pts2)
    canvas = np.hstack([left, right])
    cv2.imwrite(str(out_path), canvas)
