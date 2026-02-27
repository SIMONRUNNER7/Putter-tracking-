"""
Golf-specific augmentation pipeline.

Takes the auto-labeled dataset and generates augmented copies that simulate:
  - Real-course lighting (shadows, highlights, time-of-day variation)
  - Motion blur (camera or putter movement)
  - Different green textures (rough, fringe, artificial turf)
  - Camera shake / slight perspective tilt
  - Partial occlusion (player's feet, shadows, tee markers)
  - Reflections on metallic putter surfaces

Each augmentation preserves YOLO bounding box labels exactly
(or adjusts them for geometric transforms).

Usage:
    python training/augment.py \
        --dataset data/putter_dataset \
        --factor  5          # generate 5x augmented copies per image
        --backgrounds data/green_backgrounds   # optional real green images

Full pipeline:
    1. python training/scraper.py   scrape  --out data/raw_images --limit 300
    2. python training/autolabel.py label   --images data/raw_images --out data/putter_dataset
    3. python training/autolabel.py split   --dataset data/putter_dataset
    4. python training/augment.py           --dataset data/putter_dataset --factor 5
    5. python training/train_detector.py    train --data data/putter_dataset/data.yaml
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Individual augmentation functions
# Each takes (bgr, bbox_list) and returns (augmented_bgr, new_bbox_list)
# bbox_list = list of (cx, cy, w, h) in normalised [0,1] coords
# ---------------------------------------------------------------------------

BBoxes = list[tuple[float, float, float, float]]


def aug_motion_blur(bgr: np.ndarray, bboxes: BBoxes, strength: int = 0) -> tuple:
    """Simulate camera or putter motion blur."""
    if strength == 0:
        strength = random.randint(3, 12)
    # Random direction
    angle  = random.uniform(0, 180)
    kernel = _motion_kernel(strength, angle)
    blurred = cv2.filter2D(bgr, -1, kernel)
    return blurred, bboxes   # bboxes unchanged


def aug_shadow(bgr: np.ndarray, bboxes: BBoxes) -> tuple:
    """Add a realistic shadow polygon across the image."""
    h, w = bgr.shape[:2]
    overlay = bgr.copy().astype(np.float32)

    num_pts = random.randint(4, 7)
    pts = np.array([
        [random.randint(0, w), random.randint(0, h)]
        for _ in range(num_pts)
    ], dtype=np.int32)

    darkness = random.uniform(0.35, 0.65)
    shadow_layer = np.ones_like(overlay)
    cv2.fillPoly(shadow_layer, [pts], (darkness, darkness, darkness))

    result = (overlay * shadow_layer).clip(0, 255).astype(np.uint8)
    return result, bboxes


def aug_brightness(bgr: np.ndarray, bboxes: BBoxes) -> tuple:
    """Random brightness + contrast adjustment."""
    alpha = random.uniform(0.6, 1.4)   # contrast
    beta  = random.randint(-40, 40)    # brightness
    result = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
    return result, bboxes


def aug_hue_shift(bgr: np.ndarray, bboxes: BBoxes) -> tuple:
    """Shift hue (simulates different lighting colour temperature)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-15, 15)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-30, 30), 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result, bboxes


def aug_noise(bgr: np.ndarray, bboxes: BBoxes) -> tuple:
    """Add Gaussian noise (simulates low-quality camera sensor)."""
    sigma = random.uniform(5, 25)
    noise = np.random.normal(0, sigma, bgr.shape).astype(np.int16)
    result = np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return result, bboxes


def aug_flip_horizontal(bgr: np.ndarray, bboxes: BBoxes) -> tuple:
    """Horizontal flip (left-handed vs right-handed golfer)."""
    result = cv2.flip(bgr, 1)
    new_bboxes = [(1.0 - cx, cy, bw, bh) for cx, cy, bw, bh in bboxes]
    return result, new_bboxes


def aug_rotate(bgr: np.ndarray, bboxes: BBoxes, max_angle: float = 8.0) -> tuple:
    """Small rotation (camera tilt, uneven ground)."""
    h, w = bgr.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    result = cv2.warpAffine(bgr, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)

    # Transform bboxes
    new_bboxes = []
    for cx, cy, bw, bh in bboxes:
        # Convert normalised → pixel corners
        corners = np.array([
            [(cx - bw/2)*w, (cy - bh/2)*h],
            [(cx + bw/2)*w, (cy - bh/2)*h],
            [(cx + bw/2)*w, (cy + bh/2)*h],
            [(cx - bw/2)*w, (cy + bh/2)*h],
        ])
        ones   = np.ones((4, 1))
        pts_h  = np.hstack([corners, ones])
        rotated = (M @ pts_h.T).T

        x_min = max(0, rotated[:, 0].min()) / w
        x_max = min(w, rotated[:, 0].max()) / w
        y_min = max(0, rotated[:, 1].min()) / h
        y_max = min(h, rotated[:, 1].max()) / h

        new_cx = (x_min + x_max) / 2
        new_cy = (y_min + y_max) / 2
        new_bw = x_max - x_min
        new_bh = y_max - y_min
        new_bboxes.append((new_cx, new_cy, new_bw, new_bh))

    return result, new_bboxes


def aug_perspective(bgr: np.ndarray, bboxes: BBoxes, strength: float = 0.05) -> tuple:
    """Slight perspective warp (angled camera, different lie angle)."""
    h, w = bgr.shape[:2]
    s = strength * min(h, w)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.uniform(-s, s), random.uniform(-s, s)],
        [w + random.uniform(-s, s), random.uniform(-s, s)],
        [w + random.uniform(-s, s), h + random.uniform(-s, s)],
        [random.uniform(-s, s), h + random.uniform(-s, s)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(bgr, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)

    # Transform bboxes (approximate: transform centre point only)
    new_bboxes = []
    for cx, cy, bw, bh in bboxes:
        pt = np.array([[[cx*w, cy*h]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, M)
        new_cx = float(transformed[0, 0, 0]) / w
        new_cy = float(transformed[0, 0, 1]) / h
        # bbox size changes slightly — scale by warp magnitude
        new_bboxes.append((
            np.clip(new_cx, 0.01, 0.99),
            np.clip(new_cy, 0.01, 0.99),
            bw, bh  # approximate: size unchanged
        ))
    return result, new_bboxes


def aug_background_swap(
    bgr: np.ndarray,
    bboxes: BBoxes,
    bg_images: list[np.ndarray],
) -> tuple:
    """
    Composite the putter onto a different green background.
    Requires the original image to have a reasonably clean background.
    """
    if not bg_images:
        return bgr, bboxes

    h, w = bgr.shape[:2]
    bg = random.choice(bg_images)
    bg = cv2.resize(bg, (w, h))

    # Simple background removal: segment by green/white
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Green bg
    green_mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
    # White bg
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
    bg_mask = cv2.bitwise_or(green_mask, white_mask)

    # Smooth the mask edge
    bg_mask_smooth = cv2.GaussianBlur(bg_mask, (9, 9), 0)
    alpha = (bg_mask_smooth / 255.0)[..., np.newaxis]

    # Blend: background where bg_mask, putter where ~bg_mask
    result = (bg.astype(np.float32) * alpha +
              bgr.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return result, bboxes


def aug_reflection(bgr: np.ndarray, bboxes: BBoxes) -> tuple:
    """Add a subtle metallic sheen / highlight on the putter head."""
    h, w = bgr.shape[:2]
    overlay = bgr.copy().astype(np.float32)

    # Random ellipse highlight
    for bbox in bboxes:
        cx, cy, bw, bh = bbox
        px = int(cx * w)
        py = int(cy * h)
        axes = (int(bw * w * 0.15), int(bh * h * 0.08))
        angle = random.randint(0, 180)
        intensity = random.uniform(0.6, 1.0)

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (px, py), axes, angle, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., np.newaxis]
        overlay = overlay + mask * intensity * 60
    return overlay.clip(0, 255).astype(np.uint8), bboxes


# ---------------------------------------------------------------------------
# Augmentation sequence
# ---------------------------------------------------------------------------

# Available augmentations with their probability of being applied
AUG_MENU = [
    (aug_motion_blur,      0.40),
    (aug_shadow,           0.50),
    (aug_brightness,       0.70),
    (aug_hue_shift,        0.60),
    (aug_noise,            0.35),
    (aug_flip_horizontal,  0.50),
    (aug_rotate,           0.45),
    (aug_perspective,      0.30),
    (aug_reflection,       0.25),
]


def augment_sample(
    bgr: np.ndarray,
    bboxes: BBoxes,
    bg_images: Optional[list] = None,
    n_augs: int = 3,
) -> tuple[np.ndarray, BBoxes]:
    """Apply n_augs randomly selected augmentations to one image."""
    aug_funcs = [
        fn for fn, prob in AUG_MENU
        if random.random() < prob
    ]
    random.shuffle(aug_funcs)
    aug_funcs = aug_funcs[:n_augs]

    result, cur_bboxes = bgr.copy(), list(bboxes)

    for fn in aug_funcs:
        if fn == aug_background_swap and bg_images:
            result, cur_bboxes = fn(result, cur_bboxes, bg_images)
        else:
            try:
                result, cur_bboxes = fn(result, cur_bboxes)
            except Exception:
                pass   # skip failed augmentation

    return result, cur_bboxes


# ---------------------------------------------------------------------------
# Dataset augmentor
# ---------------------------------------------------------------------------

def augment_dataset(
    dataset_dir: str,
    factor: int = 5,
    bg_dir: Optional[str] = None,
    split: str = "train",   # augment only training split
) -> None:
    """
    Augment all images in dataset_dir/images/{split}/
    by `factor` times, saving to the same directory.
    """
    dst      = Path(dataset_dir)
    img_dir  = dst / "images" / split
    lbl_dir  = dst / "labels" / split

    if not img_dir.exists():
        print(f"[augment] {img_dir} not found — run autolabel + split first.")
        return

    # Load background images if provided
    bg_images: list[np.ndarray] = []
    if bg_dir:
        for p in Path(bg_dir).glob("*.jpg"):
            img = cv2.imread(str(p))
            if img is not None:
                bg_images.append(img)
        for p in Path(bg_dir).glob("*.png"):
            img = cv2.imread(str(p))
            if img is not None:
                bg_images.append(img)
        print(f"[augment] Loaded {len(bg_images)} background images.")

    originals = list(img_dir.glob("*.jpg"))
    print(f"[augment] Augmenting {len(originals)} images × {factor} = "
          f"{len(originals) * factor} new samples…")

    for img_path in tqdm(originals, desc="Augmenting", unit="img"):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        bboxes = _read_yolo_labels(lbl_path)
        if not bboxes:
            continue

        for i in range(factor):
            aug_bgr, aug_bboxes = augment_sample(
                bgr, bboxes, bg_images, n_augs=random.randint(2, 4)
            )
            stem     = f"{img_path.stem}_aug{i:03d}"
            out_img  = img_dir  / (stem + ".jpg")
            out_lbl  = lbl_dir  / (stem + ".txt")

            cv2.imwrite(str(out_img), aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            _write_yolo_labels(out_lbl, aug_bboxes)

    new_count = len(list(img_dir.glob("*.jpg")))
    print(f"[augment] Done. Training set now has {new_count} images.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _motion_kernel(size: int, angle: float) -> np.ndarray:
    """Create a directional motion blur kernel."""
    k = np.zeros((size, size))
    cx = size // 2
    angle_rad = np.radians(angle)
    for i in range(size):
        offset = i - cx
        x = cx + int(round(offset * np.cos(angle_rad)))
        y = cx + int(round(offset * np.sin(angle_rad)))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1
    s = k.sum()
    return k / s if s > 0 else k


def _read_yolo_labels(path: Path) -> BBoxes:
    bboxes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, cx, cy, bw, bh = parts[:5]
                bboxes.append((float(cx), float(cy), float(bw), float(bh)))
    return bboxes


def _write_yolo_labels(path: Path, bboxes: BBoxes) -> None:
    with open(path, "w") as f:
        for cx, cy, bw, bh in bboxes:
            # Clip to valid range
            cx = np.clip(cx, 0.001, 0.999)
            cy = np.clip(cy, 0.001, 0.999)
            bw = np.clip(bw, 0.001, 1.0)
            bh = np.clip(bh, 0.001, 1.0)
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Augment putter dataset for robust YOLO training"
    )
    parser.add_argument("--dataset",     default="data/putter_dataset")
    parser.add_argument("--factor",      type=int,   default=5,
                        help="Augmentation multiplier (default: 5)")
    parser.add_argument("--backgrounds", default=None,
                        help="Directory of green background images (optional)")
    parser.add_argument("--split",       default="train",
                        help="Which split to augment (default: train)")

    args = parser.parse_args()
    augment_dataset(
        dataset_dir=args.dataset,
        factor=args.factor,
        bg_dir=args.backgrounds,
        split=args.split,
    )


if __name__ == "__main__":
    main()
