#!/usr/bin/env python3
"""
Quick test of the trained YOLO putter detector.

Usage:
    # Test on validation images (default)
    python test_model.py

    # Test on a video file
    python test_model.py path/to/putt.mp4

    # Custom model path
    python test_model.py --model runs/detect/runs/putter/putter_detector/weights/best.pt video.mp4
"""
import sys
import os
import argparse
import glob

MODEL_PATH = "runs/detect/runs/putter/putter_detector/weights/best.pt"
VAL_DIR    = "data/putter_dataset/images/val"
OUTPUT_DIR = "runs/test_output"


def run_on_images(model, output_dir: str) -> None:
    images = sorted(glob.glob(os.path.join(VAL_DIR, "*.jpg")))
    if not images:
        print(f"[ERROR] No images found in {VAL_DIR}")
        return

    print(f"[test] Running on {len(images)} validation images → {output_dir}")
    results = model(images, save=True, project=output_dir, name="images",
                    conf=0.25, iou=0.45, exist_ok=True)

    detected = sum(1 for r in results if len(r.boxes) > 0)
    print(f"[test] Detections: {detected}/{len(images)} images")

    for r in results[:5]:   # show first 5
        fname = os.path.basename(r.path)
        if len(r.boxes) > 0:
            conf = float(r.boxes.conf[0])
            box  = r.boxes.xyxy[0].tolist()
            print(f"  ✓ {fname}  conf={conf:.2f}  box={[round(v) for v in box]}")
        else:
            print(f"  ✗ {fname}  no detection")


def run_on_video(model, video_path: str, output_dir: str) -> None:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "output_" + os.path.basename(video_path))
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    print(f"[test] Video: {w}x{h} @ {fps:.0f}fps  ({total} frames)")
    print(f"[test] Output → {out_path}")

    frame_idx   = 0
    det_count   = 0
    conf_sum    = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False, conf=0.25)[0]

        # Draw detections
        annotated = results.plot()

        if len(results.boxes) > 0:
            det_count += 1
            conf_sum  += float(results.boxes.conf[0])

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 30 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            avg_conf = conf_sum / det_count if det_count > 0 else 0
            print(f"  frame {frame_idx}/{total} ({pct:.0f}%)  "
                  f"detections={det_count}  avg_conf={avg_conf:.2f}")

    cap.release()
    writer.release()

    avg_conf = conf_sum / det_count if det_count > 0 else 0
    print(f"\n[test] Done — {det_count}/{frame_idx} frames detected  "
          f"avg_conf={avg_conf:.2f}")
    print(f"[test] Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", help="Path to video file (optional)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to best.pt")
    parser.add_argument("--out",   default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("  → Run training first, or specify --model path/to/best.pt")
        sys.exit(1)

    print(f"[test] Loading model: {args.model}")
    from ultralytics import YOLO
    model = YOLO(args.model)

    os.makedirs(args.out, exist_ok=True)

    if args.video:
        if not os.path.isfile(args.video):
            print(f"[ERROR] Video not found: {args.video}")
            sys.exit(1)
        run_on_video(model, args.video, args.out)
    else:
        run_on_images(model, args.out)


if __name__ == "__main__":
    main()
