"""Stage A — metric calibration from a circular reference object.

A circular object of known real-world diameter (a plate in the app, a coin in the
ECUSTFD benchmark) projects to an *ellipse* under perspective. Fitting that ellipse
and reading its **major axis** — the diameter that is *not* foreshortened by camera
tilt — gives a robust centimetres-per-pixel scale.

The module is deliberately classical (OpenCV only). Calibration must be robust and
explainable, and the geometry here is fully understood, so there is nothing to learn.

Typical use::

    from foodvol.calibration import calibrate
    calib = calibrate(bgr_image, real_diameter_cm=26.0, expect="largest")  # a plate
    area_cm2 = calib.pixel_area_to_cm2(mask.sum())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Calibration:
    """Result of a calibration: how to convert pixels to centimetres.

    Attributes
    ----------
    cm_per_px : metres-free linear scale; multiply a pixel length to get centimetres.
    px_per_cm : inverse of ``cm_per_px``.
    center    : (x, y) pixel centre of the detected reference ellipse.
    axes      : (major, minor) full axis lengths in pixels.
    angle_deg : rotation of the ellipse in degrees (OpenCV convention).
    method    : which detector produced the result ("ellipse" or "hough").
    score     : detector confidence in [0, 1]; higher is better.
    """

    cm_per_px: float
    px_per_cm: float
    center: tuple[float, float]
    axes: tuple[float, float]
    angle_deg: float
    method: str
    score: float = field(default=1.0)

    # --- unit conversions ------------------------------------------------------
    def pixel_length_to_cm(self, n_pixels: float) -> float:
        """Convert a length in pixels to centimetres."""
        return float(n_pixels) * self.cm_per_px

    def pixel_area_to_cm2(self, n_pixels: float) -> float:
        """Convert an area in pixels to square centimetres."""
        return float(n_pixels) * (self.cm_per_px ** 2)

    @property
    def diameter_px(self) -> float:
        """Reference diameter in pixels (the ellipse major axis)."""
        return max(self.axes)

    def interior_mask(self, image_shape: tuple[int, int]) -> np.ndarray:
        """Boolean mask of the reference ellipse interior.

        For a plate this is the plate surface, used downstream to restrict food
        segmentation to the inside of the plate.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = self.center
        major, minor = self.axes
        cv2.ellipse(
            mask,
            (int(round(cx)), int(round(cy))),
            (int(round(major / 2)), int(round(minor / 2))),
            self.angle_deg, 0, 360, color=255, thickness=-1,
        )
        return mask.astype(bool)


@dataclass
class _Candidate:
    center: tuple[float, float]
    major: float          # full major-axis length (px)
    minor: float          # full minor-axis length (px)
    angle: float
    method: str
    score: float

    @property
    def area(self) -> float:
        return np.pi / 4.0 * self.major * self.minor


class CalibrationError(RuntimeError):
    """Raised when no reference object could be detected."""


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _ellipse_candidates(gray: np.ndarray, min_axis_px: float) -> list[_Candidate]:
    """Fit ellipses to strong external contours (robust for large objects/plates)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out: list[_Candidate] = []
    for cnt in contours:
        if len(cnt) < 5:                      # fitEllipse needs >= 5 points
            continue
        (cx, cy), (ax1, ax2), angle = cv2.fitEllipse(cnt)
        major, minor = max(ax1, ax2), min(ax1, ax2)
        if minor < min_axis_px or major <= 0:
            continue
        # Support: how well the contour fills its fitted ellipse (1.0 == perfect).
        ellipse_area = np.pi / 4.0 * major * minor
        contour_area = cv2.contourArea(cnt)
        support = float(np.clip(contour_area / max(ellipse_area, 1e-6), 0.0, 1.0))
        roundness = minor / major             # 1.0 == circle (top-down), lower == tilted
        score = 0.6 * support + 0.4 * roundness
        out.append(_Candidate((cx, cy), major, minor, angle, "ellipse", score))
    return out


def _hough_candidates(gray: np.ndarray, min_r: int, max_r: int) -> list[_Candidate]:
    """Detect circles via the Hough transform (robust for small objects/coins)."""
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(min_r, 10),
        param1=120, param2=40, minRadius=int(min_r), maxRadius=int(max_r),
    )
    out: list[_Candidate] = []
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(float):
            out.append(_Candidate((cx, cy), 2 * r, 2 * r, 0.0, "hough", 0.7))
    return out


def calibrate(
    image: np.ndarray,
    real_diameter_cm: float,
    expect: str = "largest",
    roi: Optional[tuple[int, int, int, int]] = None,
) -> Calibration:
    """Detect the circular reference and return a :class:`Calibration`.

    Parameters
    ----------
    image : BGR (OpenCV) or grayscale image.
    real_diameter_cm : the reference object's true diameter in centimetres
        (e.g. the plate diameter entered by the user, or 2.5 for the ECUSTFD coin).
    expect : ``"largest"`` selects the biggest detected circle (a plate),
        ``"smallest"`` the smallest plausible one (a coin).
    roi : optional ``(x, y, w, h)`` to restrict the search region.

    Raises
    ------
    CalibrationError : if no reference object is found.
    """
    if real_diameter_cm <= 0:
        raise ValueError("real_diameter_cm must be positive")

    gray = _to_gray(image)
    ox, oy = 0, 0
    if roi is not None:
        ox, oy, rw, rh = roi
        gray = gray[oy:oy + rh, ox:ox + rw]

    h, w = gray.shape[:2]
    short_side = min(h, w)

    if expect == "largest":              # plate: big, dominant
        min_axis = 0.10 * short_side
        candidates = _ellipse_candidates(gray, min_axis)
        candidates += _hough_candidates(gray, int(0.10 * short_side), int(0.60 * short_side))
    elif expect == "smallest":           # coin: small, circular
        min_axis = max(8.0, 0.01 * short_side)
        candidates = _hough_candidates(gray, max(5, int(0.01 * short_side)), int(0.12 * short_side))
        candidates += [c for c in _ellipse_candidates(gray, min_axis)
                       if c.minor / c.major > 0.7]   # coins stay near-circular
    else:
        raise ValueError("expect must be 'largest' or 'smallest'")

    candidates = [c for c in candidates if c.minor >= min_axis]
    if not candidates:
        raise CalibrationError(
            f"No circular reference detected (expect={expect!r}). "
            "Ensure the plate/coin is fully visible and well lit."
        )

    if expect == "largest":
        # Biggest object, lightly tie-broken by detector score.
        pick = max(candidates, key=lambda c: (c.area, c.score))
    else:
        # Smallest plausible object, tie-broken by score.
        pick = min(candidates, key=lambda c: (c.area, -c.score))

    diameter_px = pick.major
    cm_per_px = real_diameter_cm / diameter_px
    return Calibration(
        cm_per_px=cm_per_px,
        px_per_cm=1.0 / cm_per_px,
        center=(pick.center[0] + ox, pick.center[1] + oy),
        axes=(pick.major, pick.minor),
        angle_deg=pick.angle,
        method=pick.method,
        score=float(np.clip(pick.score, 0.0, 1.0)),
    )


def calibrate_manual(diameter_px: float, real_diameter_cm: float,
                     center: tuple[float, float] = (0.0, 0.0)) -> Calibration:
    """Build a :class:`Calibration` directly from a known pixel diameter.

    Useful as a fallback when automatic detection fails but the user can mark the
    plate diameter in the image, or for unit tests.
    """
    cm_per_px = real_diameter_cm / float(diameter_px)
    return Calibration(
        cm_per_px=cm_per_px, px_per_cm=1.0 / cm_per_px, center=center,
        axes=(float(diameter_px), float(diameter_px)), angle_deg=0.0,
        method="manual", score=1.0,
    )
