"""
Multi-method putter detection pipeline.

Strategy cascade (each frame):
  1. Motion mask   – Background subtraction (MOG2)  → isolates moving object
  2. Line detect   – Hough Transform on edge image  → finds shaft
  3. Shape/contour – Contour analysis on motion mask → finds head
  4. Optical flow  – Lucas-Kanade on tracked corners → refines position
  5. Template match– ORB feature matching            → last resort
  6. Kalman filter – Smoothing + prediction if all else fails

This cascade means the tracker degrades gracefully under:
  - Poor lighting (CLAHE pre-processing)
  - Shadows (motion-based cues survive shadows)
  - Camera shake (Kalman prediction)
  - Partial occlusion (optical flow carries through)
"""
from __future__ import annotations

import cv2
import numpy as np
import math
from typing import Optional
from dataclasses import dataclass
from utils.geometry import KalmanTracker2D
from utils.drawing import preprocess_for_detection


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    head_x: float
    head_y: float
    shaft_start: Optional[tuple] = None
    shaft_end:   Optional[tuple] = None
    face_angle:  Optional[float] = None   # degrees
    shaft_angle: Optional[float] = None   # degrees
    confidence:  float = 0.0
    method: str = "none"


# ---------------------------------------------------------------------------
# Putter Detector
# ---------------------------------------------------------------------------

class PutterDetector:
    """
    Full detection pipeline.  Call detect(frame) each frame.
    On the first frame, call initialize() so the background model can warm up.
    """

    def __init__(
        self,
        target_line_angle: float = 0.0,
        use_yolo: bool = False,
        yolo_model_path: Optional[str] = None,
    ):
        self.target_line_angle = target_line_angle  # degrees (screen angle)

        # Background subtractor (MOG2 is more robust to gradual light changes)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=40,
            detectShadows=True,    # shadow suppression
        )

        # Optical flow state
        self._of_pts: Optional[np.ndarray] = None   # tracked corner points
        self._prev_gray: Optional[np.ndarray] = None
        self._of_mask: Optional[np.ndarray] = None  # region of interest mask

        # Kalman filter for head position
        self._kalman = KalmanTracker2D(process_noise=2.0, measurement_noise=8.0)
        self._kalman_active = False

        # Template (set when user clicks on putter)
        self._template: Optional[np.ndarray] = None
        self._template_kp = None
        self._template_desc = None
        self._orb = cv2.ORB_create(nfeatures=500)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # YOLO (optional)
        self._use_yolo = use_yolo and yolo_model_path is not None
        self._yolo = None
        if self._use_yolo:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(yolo_model_path)
            except Exception as e:
                print(f"[PutterDetector] YOLO load failed: {e}")
                self._use_yolo = False

        # Consecutive frames with no detection
        self._lost_frames = 0
        self._last_detection: Optional[Detection] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initialize_background(self, frames: list[np.ndarray]) -> None:
        """Warm up the background model with static frames (putter out of view)."""
        for f in frames:
            self.bg_subtractor.apply(preprocess_for_detection(f))

    def set_template(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """
        Set a putter head template from a user-defined bounding box.
        bbox = (x, y, w, h)
        """
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return
        self._template = roi.copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is not None and len(kp) > 0:
            self._template_kp = kp
            self._template_desc = desc

        # Also seed Kalman
        self._kalman.init(x + w/2, y + h/2)
        self._kalman_active = True
        self._kalman.update(x + w/2, y + h/2)

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> Detection:
        """
        Main detection call.  Returns the best Detection for this frame.
        """
        enhanced = preprocess_for_detection(frame)
        gray     = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # Kalman prediction first (provides prior)
        if self._kalman_active:
            px, py = self._kalman.predict()
        else:
            px, py = frame.shape[1]/2, frame.shape[0]/2

        best: Optional[Detection] = None

        # ---- Priority 1: YOLO (if available and loaded) -----------------
        if self._use_yolo and self._yolo is not None:
            best = self._detect_yolo(frame)

        # ---- Priority 2: Motion + Line (most frames during a putt) ------
        if best is None or best.confidence < 0.5:
            motion_det = self._detect_motion_and_lines(enhanced, gray)
            if motion_det and (best is None or motion_det.confidence > best.confidence):
                best = motion_det

        # ---- Priority 3: Optical flow refinement ------------------------
        if self._prev_gray is not None and self._of_pts is not None:
            of_det = self._detect_optical_flow(gray, px, py)
            if of_det and (best is None or of_det.confidence > best.confidence * 0.9):
                # Optical flow blends with motion detection
                if best is not None and best.confidence > 0.4:
                    # Weighted average
                    w1, w2 = best.confidence, of_det.confidence
                    total = w1 + w2
                    best.head_x = (best.head_x * w1 + of_det.head_x * w2) / total
                    best.head_y = (best.head_y * w1 + of_det.head_y * w2) / total
                elif of_det is not None:
                    best = of_det

        # ---- Priority 4: Template matching (fallback) -------------------
        if (best is None or best.confidence < 0.25) and self._template is not None:
            tmpl_det = self._detect_template(gray, px, py)
            if tmpl_det and (best is None or tmpl_det.confidence > best.confidence):
                best = tmpl_det

        # ---- Priority 5: Kalman prediction only -------------------------
        if best is None or best.confidence < 0.15:
            if self._kalman_active:
                best = Detection(
                    head_x=px, head_y=py,
                    confidence=max(0.0, 0.15 - self._lost_frames * 0.02),
                    method="kalman_pred",
                )

        # Update Kalman & optical flow state
        if best is not None and best.confidence > 0.2:
            self._kalman.update(best.head_x, best.head_y)
            self._kalman_active = True
            self._lost_frames = 0
            self._update_optical_flow_points(gray, best)
        else:
            self._lost_frames += 1

        # Store for next frame
        self._prev_gray = gray.copy()
        self._last_detection = best

        return best or Detection(head_x=px, head_y=py, confidence=0.0, method="none")

    # ------------------------------------------------------------------
    # Private detection methods
    # ------------------------------------------------------------------

    def _detect_motion_and_lines(
        self, enhanced: np.ndarray, gray: np.ndarray
    ) -> Optional[Detection]:
        """
        Strategy: MOG2 background subtraction → find moving region →
        apply Hough Lines to find shaft → head at the lower end of shaft.
        """
        h, w = enhanced.shape[:2]

        # --- Motion mask
        fg_mask = self.bg_subtractor.apply(enhanced)
        # Remove shadows (they are labeled 127, foreground is 255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel7)
        fg_mask = cv2.dilate(fg_mask, kernel7, iterations=1)

        if cv2.countNonZero(fg_mask) < 100:
            return None   # Nothing moving

        # --- Edge detection within motion region only
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.bitwise_and(edges, fg_mask)

        # --- Hough Line Transform to find shaft
        lines = cv2.HoughLinesP(
            edges,
            rho=1, theta=np.pi/180,
            threshold=25,
            minLineLength=60,   # shaft must be long
            maxLineGap=15,
        )

        shaft_start, shaft_end, shaft_angle = None, None, None
        head_x, head_y = None, None
        confidence = 0.0

        if lines is not None:
            # Pick the longest line (most likely the shaft)
            best_len = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                if length > best_len:
                    best_len = length
                    shaft_start = (float(x1), float(y1))
                    shaft_end   = (float(x2), float(y2))
                    shaft_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            if shaft_start and shaft_end:
                # Head is at the end of the shaft closest to the ground
                # (larger y = lower in frame = closer to ground)
                if shaft_start[1] > shaft_end[1]:
                    head_x, head_y = shaft_start
                else:
                    head_x, head_y = shaft_end

                confidence = min(1.0, best_len / 200.0) * 0.7

        # --- If no good shaft, fall back to largest motion blob centroid
        if head_x is None:
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 200:
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        head_x = M["m10"] / M["m00"]
                        head_y = M["m01"] / M["m00"]
                        confidence = min(0.4, area / 5000.0)

        if head_x is None:
            return None

        # --- Face angle from shaft angle + target line offset
        face_angle = None
        if shaft_angle is not None:
            face_angle = _shaft_to_face_angle(shaft_angle, self.target_line_angle)

        return Detection(
            head_x=head_x, head_y=head_y,
            shaft_start=shaft_start, shaft_end=shaft_end,
            face_angle=face_angle, shaft_angle=shaft_angle,
            confidence=confidence,
            method="motion+hough",
        )

    def _detect_optical_flow(
        self, gray: np.ndarray, pred_x: float, pred_y: float
    ) -> Optional[Detection]:
        """
        Lucas-Kanade optical flow on previously detected feature points.
        """
        if self._of_pts is None or len(self._of_pts) < 3:
            return None

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._of_pts, None, **lk_params
        )

        good_new = new_pts[status.ravel() == 1]
        if len(good_new) < 3:
            self._of_pts = None
            return None

        # Centroid of tracked points = head position estimate
        cx = float(np.mean(good_new[:, 0, 0]))
        cy = float(np.mean(good_new[:, 0, 1]))

        # Update tracked points for next frame
        self._of_pts = good_new.reshape(-1, 1, 2)

        confidence = min(0.85, len(good_new) / 20.0)
        return Detection(
            head_x=cx, head_y=cy,
            confidence=confidence,
            method="optical_flow",
        )

    def _detect_template(
        self, gray: np.ndarray, pred_x: float, pred_y: float
    ) -> Optional[Detection]:
        """ORB feature matching against the stored putter template."""
        if self._template_desc is None:
            return None

        kp2, desc2 = self._orb.detectAndCompute(gray, None)
        if desc2 is None or len(kp2) < 4:
            return None

        matches = self._bf.match(self._template_desc, desc2)
        if len(matches) < 4:
            return None

        # Best matches → homography → estimate head position
        matches = sorted(matches, key=lambda m: m.distance)[:20]
        src_pts = np.float32([self._template_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return None

        # Transform template center
        th, tw = self._template.shape[:2]
        center = np.array([[tw/2, th/2]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(center, M)
        cx = float(transformed[0, 0, 0])
        cy = float(transformed[0, 0, 1])

        inliers = int(np.sum(mask)) if mask is not None else 0
        confidence = min(0.6, inliers / 15.0)

        return Detection(
            head_x=cx, head_y=cy,
            confidence=confidence,
            method="template_orb",
        )

    def _detect_yolo(self, frame: np.ndarray) -> Optional[Detection]:
        """YOLO detection (if custom model is available)."""
        try:
            results = self._yolo(frame, verbose=False)[0]
            best_conf = 0.0
            best_box = None
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()

            if best_box is None:
                return None

            x1, y1, x2, y2 = best_box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            return Detection(
                head_x=cx, head_y=cy,
                confidence=float(best_conf),
                method="yolo",
            )
        except Exception:
            return None

    def _update_optical_flow_points(
        self, gray: np.ndarray, detection: Detection
    ) -> None:
        """Extract Shi-Tomasi corners around detected head for optical flow."""
        cx, cy = int(detection.head_x), int(detection.head_y)
        h, w = gray.shape
        roi_half = 50
        x0 = max(0, cx - roi_half)
        y0 = max(0, cy - roi_half)
        x1 = min(w, cx + roi_half)
        y1 = min(h, cy + roi_half)

        roi = gray[y0:y1, x0:x1]
        corners = cv2.goodFeaturesToTrack(
            roi, maxCorners=30, qualityLevel=0.01, minDistance=5
        )
        if corners is not None and len(corners) >= 3:
            corners[:, 0, 0] += x0
            corners[:, 0, 1] += y0
            self._of_pts = corners.astype(np.float32)
        else:
            self._of_pts = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shaft_to_face_angle(shaft_angle_deg: float, target_line_deg: float) -> float:
    """
    The putter face is perpendicular to the shaft.
    Face angle relative to the target line = shaft_angle + 90 - target_line.
    Returned in [-90, 90] degrees.
    """
    face_direction = shaft_angle_deg + 90.0
    angle = face_direction - target_line_deg
    # Normalize to [-90, 90]
    angle = (angle + 90) % 180 - 90
    return round(angle, 2)


def build_roi_mask(shape: tuple[int, int], cx: float, cy: float, radius: int) -> np.ndarray:
    """Create a circular mask around (cx, cy) with given radius."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
    return mask
