"""Streamlit demo: estimate food mass, calories and macros from plate photos or a video.

Run with::

    streamlit run app.py

The user provides either two photos (top-down + optional side) or a short video that
pans around the plate, and enters the plate diameter. The app runs the
:class:`foodvol.pipeline.FoodVolumePipeline` and shows per-item mass / calories /
macros with an annotated overlay.
"""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from foodvol import config
from foodvol.pipeline import FoodVolumePipeline, PlateEstimate
from foodvol.video import extract_frames, select_views

st.set_page_config(page_title="Food Volume & Calorie Estimator", page_icon="🍽️", layout="wide")

# Distinct overlay colours (BGR) cycled per detected item.
_PALETTE = [(60, 60, 255), (60, 200, 60), (255, 160, 0), (200, 60, 200),
            (0, 200, 200), (160, 120, 60), (60, 160, 255)]


@st.cache_resource(show_spinner="Loading models (first run downloads weights)…")
def get_pipeline() -> FoodVolumePipeline:
    return FoodVolumePipeline()


def _read_upload(uploaded) -> np.ndarray:
    """Decode an uploaded image to a BGR array."""
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@st.cache_data(show_spinner="Selecting top & side frames from the video…")
def _views_from_video(video_bytes: bytes, suffix: str):
    """Extract frames and pick a top + side view. Cached on the raw video bytes."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(video_bytes)
        path = tf.name
    try:
        frames = extract_frames(path, max_frames=24)
        # Plate diameter does not affect view *selection* (roundness is scale-free).
        return select_views(frames, plate_diameter_cm=26.0)
    finally:
        os.unlink(path)


def _rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _overlay(top_bgr: np.ndarray, result: PlateEstimate) -> np.ndarray:
    """Draw each item's mask outline + label."""
    vis = top_bgr.copy()
    for idx, item in enumerate(result.items):
        color = _PALETTE[idx % len(_PALETTE)]
        mask = item.mask.mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 3)
        cxf, cyf = map(int, item.mask.centroid)
        label = f"{item.food_class} {item.mass_g:.0f}g"
        cv2.putText(vis, label, (cxf - 40, cyf), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, label, (cxf - 40, cyf), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# --- Sidebar controls ----------------------------------------------------------
st.sidebar.header("Settings")
min_conf = st.sidebar.slider(
    "Min. classification confidence", 0.0, 0.9, 0.0, 0.05,
    help="Drop recognitions with score below this. 0 = keep all.",
)
st.sidebar.caption(f"Compute device: **{config.get_device()}**")
st.sidebar.caption(
    "**Scale**: the app self-calibrates from the recognised food class — no plate, "
    "coin or marker needed. Accuracy depends on the recognised class matching a "
    "typical serving size."
)

# --- Header --------------------------------------------------------------------
st.title("🍽️ Food Volume & Calorie Estimator")
st.write(
    "Recognises the food in a photo and estimates its **mass, calories and macros** "
    "from a single image — **no plate, coin or measuring marker required**. "
    "Just upload a photo (or short video) of the food."
)

# --- Input ---------------------------------------------------------------------
mode = st.radio("Input", ["Photos", "Video"], horizontal=True)
top_bgr: np.ndarray | None = None
side_bgr: np.ndarray | None = None
views = None

if mode == "Photos":
    c1, c2 = st.columns(2)
    top_file = c1.file_uploader("Top-down photo (required)", type=["jpg", "jpeg", "png"])
    side_file = c2.file_uploader("Side photo (optional)", type=["jpg", "jpeg", "png"])
    if top_file is not None:
        top_bgr = _read_upload(top_file)
    if side_file is not None:
        side_bgr = _read_upload(side_file)
else:
    video_file = st.file_uploader("Video panning around the plate",
                                  type=["mp4", "mov", "avi", "m4v"])
    if video_file is not None:
        suffix = os.path.splitext(video_file.name)[1] or ".mp4"
        views = _views_from_video(video_file.getvalue(), suffix)
        top_bgr, side_bgr = views.top_frame, views.side_frame
        if top_bgr is None:
            st.error("Could not read frames from the video.")
        else:
            st.caption(f"Picked from {views.n_frames} sampled frames "
                       f"(plate detected in {views.n_with_plate}).")
            vc1, vc2 = st.columns(2)
            cap_t = f"Top view (roundness {views.top.roundness:.2f})" if views.top else "Top view (fallback: sharpest frame)"
            vc1.image(_rgb(top_bgr), caption=cap_t, use_container_width=True)
            if side_bgr is not None and views.side is not None:
                vc2.image(_rgb(side_bgr),
                          caption=f"Side view (roundness {views.side.roundness:.2f})",
                          use_container_width=True)

# --- Run -----------------------------------------------------------------------
if top_bgr is not None and st.button("Estimate", type="primary"):
    pipe = get_pipeline()
    with st.spinner("Analysing image…"):
        result = pipe.estimate(top_bgr, side_image=side_bgr, min_confidence=min_conf)

    if not result.items:
        st.warning("No food recognised in the image. Try a clearer photo.")
        for note in result.notes:
            st.caption(note)
        st.stop()

    # Headline totals: energy + mass + macros.
    a, b, c = st.columns(3)
    a.metric("Total calories", f"{result.total_kcal:.0f} kcal")
    b.metric("Total mass", f"{result.total_mass_g:.0f} g")
    c.metric("Items detected", str(len(result.items)))
    p, k, f = st.columns(3)
    p.metric("Protein", f"{result.total_protein_g:.0f} g")
    k.metric("Carbs", f"{result.total_carbs_g:.0f} g")
    f.metric("Fat", f"{result.total_fat_g:.0f} g")

    st.image(_overlay(top_bgr, result), caption="Detected items", use_container_width=True)

    # Per-item breakdown.
    def _alts(it):
        return ", ".join(f"{lbl} {sc:.0%}" for lbl, sc in it.alternatives) or "—"
    table = pd.DataFrame([{
        "Food": it.food_class,
        "Confidence": f"{it.confidence:.0%}",
        "Also considered": _alts(it),
        "Area (cm²)": round(it.area_cm2, 1),
        "Height (cm)": round(it.height_cm, 1),
        "h source": it.height_source,
        "Volume (mL)": round(it.volume_ml, 0),
        "Mass (g)": round(it.mass_g, 0),
        "Calories (kcal)": round(it.nutrition.kcal, 0),
        "Protein (g)": round(it.nutrition.protein_g, 1),
        "Carbs (g)": round(it.nutrition.carbs_g, 1),
        "Fat (g)": round(it.nutrition.fat_g, 1),
    } for it in result.items])
    st.dataframe(table, use_container_width=True, hide_index=True)

    # Transparency: where the numbers came from and their caveats.
    if result.items:
        scales = ", ".join(f"{it.food_class}: {it.cm_per_px:.4f} cm/px ({it.scale_source})"
                           for it in result.items)
        st.caption(f"Per-item scale (self-calibrated): {scales}")
    for note in result.notes:
        st.caption("ℹ️ " + note)
    if any(it.nutrition.is_default for it in result.items):
        st.caption("⚠️ Some items used a generic nutrition fallback (class not in the table).")
    st.info(
        "Estimates are approximate. Expect ~15–30 % mass error; accuracy is highest for a "
        "single dish photographed top-down with a side view. The volume model was trained on "
        "ECUSTFD (fruit/snacks) — collect target-domain data to improve it."
    )
elif top_bgr is None:
    st.info("⬆️ Provide a top-down photo or a video to begin.")
