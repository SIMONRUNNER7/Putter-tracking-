# PutterTrack Pro – Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       VIDEO INPUT                               │
│         (file upload or live camera capture)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRE-PROCESSING                               │
│   CLAHE normalization · Bilateral filter · Shadow suppression  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PUTTER DETECTION CASCADE                        │
│                                                                 │
│  Priority 1: YOLO v8 (custom trained)  — if model available    │
│       ↓ fallback                                                │
│  Priority 2: MOG2 Background Subtraction + Hough Lines         │
│             (finds shaft as long line in motion region)         │
│       ↓ refine                                                  │
│  Priority 3: Lucas-Kanade Optical Flow                         │
│             (tracks corners frame-to-frame)                     │
│       ↓ fallback                                                │
│  Priority 4: ORB Template Matching                              │
│             (matches against user-set ROI template)             │
│       ↓ fallback                                                │
│  Priority 5: Kalman Filter prediction                           │
│             (pure prediction when detection fails)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  KALMAN SMOOTHING                               │
│  State: [x, y, vx, vy]                                        │
│  Constant-velocity model, updated each frame                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               KEY FRAME DETECTION                               │
│                                                                 │
│  ADDRESS       = last still frame before motion starts         │
│  BACKSWING     = velocity direction reverses (vx sign change)  │
│  IMPACT        = maximum speed after backswing peak            │
│  FOLLOW-THROUGH= first significant speed drop after impact     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 METRICS COMPUTATION                             │
│                                                                 │
│  Club Path     = Arc tangent at impact vs. target line         │
│  Face Angle    = Head orientation at impact vs. target line    │
│  Face-to-Path  = Face angle – Club path                        │
│  Impact Speed  = px displacement / dt (calibrated → mph)      │
│  Tempo Ratio   = Backswing time / Downswing time               │
│  Ball Launch   = 75% Face + 25% Path  (D-plane simplified)    │
│  Distance      ≈ Speed^1.8 × 0.55                              │
│  Impact Score  = Weighted quality score (0-100)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  GUI / VISUALISATION                            │
│                                                                 │
│  VideoPlayer        : Frame scrubbing, slow motion, overlays   │
│  TrajectoryView     : Top-down arc schematic (custom QPainter) │
│  AnalysisPanel      : Metrics display + key-frame navigator    │
│  ScrubBar           : Scrubber with key-frame tick markers     │
└─────────────────────────────────────────────────────────────────┘
```

## Detection Strategy Details

### Why a cascade?

No single detection method works perfectly in all conditions:

| Method              | Works well when…             | Fails when…                    |
|---------------------|------------------------------|--------------------------------|
| YOLO (trained)      | Any condition (if trained)   | No training data available     |
| MOG2 + Hough        | Tripod camera, clear shaft   | Handheld camera, dark shaft    |
| Optical flow        | Short motion, good texture   | Large fast motion, blur        |
| Template matching   | Consistent view angle        | Rotation, scale change         |
| Kalman prediction   | Short occlusions             | Long occlusions (>1 second)    |

The cascade picks the highest-confidence detection each frame.

### Lighting & Shadow Robustness

- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) normalises
  local contrast → handles backlit or unevenly lit scenes.
- **MOG2 shadow suppression** (detectShadows=True, threshold=200) marks
  shadows as gray (127) instead of foreground, then thresholds them out.
- **LAB color space** for CLAHE separates luminance from chroma → more
  stable under lighting shifts.
- **Bilateral filter** preserves edges (shaft/head boundaries) while
  smoothing shadow-noise.

### Face Angle Detection

The putter face angle is estimated from the shaft orientation:

```
face_direction = shaft_angle + 90°
face_angle_vs_target = face_direction - target_line_angle
```

This works best with a **face-on camera** positioned perpendicular to
the target line (same as GolfBoy setup).

For precise face angle from a side/angled camera, a YOLO model trained to
also predict the face orientation, or a reflective marker system, would
give better results.

## File Structure

```
PutterTrack-/
├── main.py                     ← Application entry point
├── requirements.txt
├── setup.py
├── ARCHITECTURE.md             ← This file
│
├── app/                        ← GUI layer (PyQt6)
│   ├── main_window.py          ← Main QMainWindow + layout
│   ├── video_player.py         ← Video player + scrub bar
│   ├── analysis_panel.py       ← Metrics display panel
│   └── trajectory_view.py      ← Top-down arc visualisation
│
├── tracking/                   ← Computer vision / analysis core
│   ├── putter_detector.py      ← Multi-method detection cascade
│   ├── tracker.py              ← Full video analysis pipeline
│   ├── keyframe_detector.py    ← Automatic key frame detection
│   └── metrics.py              ← Golf metrics computation
│
├── models/                     ← Data models
│   └── shot_data.py            ← ShotRecord, ShotMetrics, etc.
│
├── utils/                      ← Utilities
│   ├── drawing.py              ← OpenCV overlay drawing
│   └── geometry.py             ← Geometric math + Kalman filter
│
├── training/                   ← YOLO model training
│   └── train_detector.py       ← Training script + dataset tools
│
└── assets/
    └── styles/
        └── dark_theme.qss      ← Qt dark theme stylesheet
```

## Improving Detection Accuracy

### Short term (no ML)
1. Set a putter ROI on the first frame (toolbar button)
2. Use a tripod for static camera → better background subtraction
3. Ensure good lighting (avoid strong side lighting)
4. Use a putter with a visible contrasting shaft (white, red, etc.)

### Medium term (custom YOLO)
1. Collect 500–1000 labeled putter images
2. Label with Roboflow (free): box around entire putter
3. Train: `python training/train_detector.py train --data data.yaml --epochs 100`
4. Load model in app: Analysis → Use YOLO Model

### Long term (production quality)
- **Multi-camera** setup (face-on + down-the-line simultaneously)
- **Reflective markers** on shaft + head (highest accuracy face angle)
- **High-speed camera** (120–240 fps) for better impact analysis
- **Radar/acoustic** integration for ball speed ground truth
