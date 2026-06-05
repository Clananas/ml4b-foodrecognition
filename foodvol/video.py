"""Video input: pick a top view and a side view from a short clip.

Instead of asking the user to take two carefully framed photos, they can record a
short video that pans around the plate. We sample frames and then choose the two we
need using the calibration itself:

* the **top view** is the frame whose detected plate is *most circular* — a circle
  seen straight down stays circular, so high roundness ≈ top-down;
* the **side view** is the frame whose plate is *most foreshortened* (lowest
  roundness) — i.e. seen most from the side, which is where height is visible.

Blurry frames are discarded via a focus measure so we don't calibrate on motion blur.
This reuses :func:`foodvol.calibration.calibrate` and needs no extra model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .calibration import CalibrationError, calibrate


@dataclass
class FrameInfo:
    index: int
    roundness: float       # minor/major axis ratio of the detected plate (1.0 == circle)
    sharpness: float       # variance of the Laplacian (focus measure)
    diameter_px: float


@dataclass
class VideoViews:
    """Frames chosen from a video plus diagnostics."""

    top_frame: Optional[np.ndarray]
    side_frame: Optional[np.ndarray]
    top: Optional[FrameInfo]
    side: Optional[FrameInfo]
    n_frames: int
    n_with_plate: int


def extract_frames(video_path: str | Path, max_frames: int = 24) -> list[np.ndarray]:
    """Sample up to ``max_frames`` frames evenly across the video (BGR arrays)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    try:
        if total <= 0:                       # some containers don't report a count
            frames = []
            while len(frames) < max_frames * 4:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            step = max(1, len(frames) // max_frames)
            return frames[::step][:max_frames]

        idxs = np.linspace(0, total - 1, num=min(max_frames, total), dtype=int)
        out = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                out.append(frame)
        return out
    finally:
        cap.release()


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_views(
    frames: list[np.ndarray],
    plate_diameter_cm: float,
    sharpness_floor: float = 0.35,
) -> VideoViews:
    """Choose a top and a side frame from sampled frames.

    ``sharpness_floor`` is relative to the sharpest frame (0–1); frames below it are
    treated as too blurry to calibrate on.
    """
    if not frames:
        return VideoViews(None, None, None, None, 0, 0)

    sharps = [_sharpness(f) for f in frames]
    max_sharp = max(sharps) or 1.0

    infos: list[FrameInfo] = []
    for i, frame in enumerate(frames):
        if sharps[i] < sharpness_floor * max_sharp:
            continue
        try:
            calib = calibrate(frame, plate_diameter_cm, expect="largest")
        except CalibrationError:
            continue
        major, minor = calib.axes
        roundness = (minor / major) if major else 0.0
        infos.append(FrameInfo(i, roundness, sharps[i], calib.diameter_px))

    if not infos:
        # No plate detected anywhere: fall back to the single sharpest frame as top.
        best = int(np.argmax(sharps))
        return VideoViews(frames[best], None, None, None, len(frames), 0)

    top_info = max(infos, key=lambda fi: fi.roundness)
    side_info = min(infos, key=lambda fi: fi.roundness)
    side_frame = frames[side_info.index] if side_info.index != top_info.index else None
    side_out = side_info if side_frame is not None else None

    return VideoViews(
        top_frame=frames[top_info.index],
        side_frame=side_frame,
        top=top_info,
        side=side_out,
        n_frames=len(frames),
        n_with_plate=len(infos),
    )
