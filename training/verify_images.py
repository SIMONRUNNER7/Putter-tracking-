"""
verify_images.py  —  vérifie les images scrappées avant annotation.

Usage:
    python3 training/verify_images.py --dir data/raw_images
    python3 training/verify_images.py --dir data/raw_images --sample 40 --save
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams["toolbar"] = "None"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(img_dir: Path) -> dict:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    imgs = []
    for ext in exts:
        imgs.extend(img_dir.glob(ext))
    imgs = [p for p in imgs if not p.name.startswith("_")]

    if not imgs:
        print(f"[verify] Aucune image trouvée dans {img_dir}")
        return {}

    widths, heights, sizes = [], [], []
    formats: dict[str, int] = {}
    hashes: set[str] = set()
    dupes = 0

    for p in imgs:
        try:
            im = Image.open(p)
            widths.append(im.width)
            heights.append(im.height)
        except Exception:
            pass

        sizes.append(p.stat().st_size)
        ext = p.suffix.lower().lstrip(".")
        formats[ext] = formats.get(ext, 0) + 1

        with open(p, "rb") as f:
            h = hashlib.md5(f.read(8192)).hexdigest()
        if h in hashes:
            dupes += 1
        hashes.add(h)

    total = len(imgs)
    print(f"\n[verify] {total} images dans {img_dir}")
    if widths:
        print(f"  Résolutions : min {min(widths)}×{min(heights)}"
              f" | max {max(widths)}×{max(heights)}"
              f" | médiane {int(np.median(widths))}×{int(np.median(heights))}")
    fmt_str = "  ".join(f"{k.upper()} {v/total*100:.0f}%" for k, v in formats.items())
    print(f"  Formats      : {fmt_str}")
    avg_kb = int(np.mean(sizes) / 1024)
    max_kb = int(max(sizes) / 1024)
    print(f"  Taille       : {avg_kb} KB moyen | {max_kb} KB max")
    print(f"  Doublons MD5 : {dupes} détectés")

    return {"paths": imgs, "widths": widths, "heights": heights}


# ---------------------------------------------------------------------------
# CV filter check
# ---------------------------------------------------------------------------

def cv_check(imgs: list[Path]) -> None:
    """Re-apply the top-view heuristic and report how many would pass."""
    passed, failed = 0, []

    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            passed += 1
            continue

        # Light background
        _, thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
        h, w = img.shape
        thresh[:5, :] = thresh[-5:, :] = thresh[:, :5] = thresh[:, -5:] = 0
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Try dark background
            _, thresh2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            failed.append((999.0, p))
            continue

        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / (h * w)
        if area_ratio < 0.03:
            failed.append((0.0, p))
            continue

        x, y, bw, bh = cv2.boundingRect(largest)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 6:
            failed.append((aspect, p))
        else:
            passed += 1

    total = len(imgs)
    print(f"\n[CV check] Passeraient le filtre : {passed}/{total}")
    if failed:
        print(f"  Echoueraient : {len(failed)}")
        print("  Top suspects (ratio aspect élevé) :")
        for asp, p in sorted(failed, key=lambda x: -x[0])[:10]:
            print(f"    aspect={asp:.1f}  {p.name}")


# ---------------------------------------------------------------------------
# Grid display
# ---------------------------------------------------------------------------

def show_grid(imgs: list[Path], n: int = 40, save: bool = False, out_dir: Path = None) -> None:
    sample = random.sample(imgs, min(n, len(imgs)))
    cols = 5
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle(f"Échantillon aléatoire — {len(sample)} images", fontsize=14)
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for ax, p in zip(axes, sample):
        try:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            im.thumbnail((400, 400), Image.LANCZOS)
            ax.imshow(np.array(im))
            label = p.stem[:22] + "\n" + f"{w}×{h}"
            ax.set_title(label, fontsize=7, pad=2)
        except Exception as e:
            ax.set_title(f"Erreur\n{p.name[:20]}", fontsize=7)

    plt.tight_layout()

    if save and out_dir:
        out_path = out_dir / "_verify_grid.jpg"
        fig.savefig(str(out_path), dpi=80, bbox_inches="tight", quality=85)
        print(f"[verify] Grille sauvegardée → {out_path}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Vérifie les images scrappées")
    parser.add_argument("--dir",    default="data/raw_images",
                        help="Dossier contenant les images (défaut: data/raw_images)")
    parser.add_argument("--sample", type=int, default=40,
                        help="Nombre d'images dans la grille (défaut: 40)")
    parser.add_argument("--save",   action="store_true",
                        help="Sauvegarder la grille en JPEG")
    parser.add_argument("--no-cv",  action="store_true",
                        help="Ignorer le re-check CV filter")
    args = parser.parse_args()

    img_dir = Path(args.dir)
    if not img_dir.exists():
        print(f"[verify] Dossier introuvable : {img_dir}")
        return

    result = compute_stats(img_dir)
    if not result:
        return

    imgs = result["paths"]

    if not args.no_cv:
        cv_check(imgs)

    show_grid(imgs, n=args.sample, save=args.save, out_dir=img_dir)


if __name__ == "__main__":
    main()
