"""Tests for video frame extraction and top/side view selection."""
import cv2
import numpy as np
import pytest

from foodvol.video import extract_frames, select_views


def _frame(minor_axis, blur=False):
    img = np.full((600, 800, 3), 30, np.uint8)
    cv2.ellipse(img, (400, 300), (200, minor_axis // 2), 0, 0, 360, (210, 210, 210), -1)
    cv2.circle(img, (400, 300), 70, (50, 80, 200), -1)
    if blur:
        img = cv2.GaussianBlur(img, (31, 31), 0)
    return img


def test_select_views_picks_round_top_and_foreshortened_side():
    frames = [_frame(m) for m in (400, 340, 260, 180, 120)] + [_frame(400, blur=True)]
    views = select_views(frames, plate_diameter_cm=26.0)
    assert views.top is not None and views.side is not None
    assert views.top.index == 0           # most circular & sharp -> top
    assert views.side.index == 4          # most foreshortened -> side
    assert views.top.roundness > views.side.roundness


def test_select_views_empty():
    views = select_views([], plate_diameter_cm=26.0)
    assert views.top_frame is None and views.n_frames == 0


def test_extract_frames_roundtrip(tmp_path):
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (800, 600))
    if not writer.isOpened():
        pytest.skip("no video codec available in this environment")
    for m in np.linspace(400, 120, 30).astype(int):
        writer.write(_frame(int(m)))
    writer.release()
    frames = extract_frames(path, max_frames=8)
    assert 1 < len(frames) <= 8
    assert frames[0].shape == (600, 800, 3)
