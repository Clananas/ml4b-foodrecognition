"""Download and extract the ECUSTFD dataset (resized version).

ECUSTFD (East China University of Science and Technology Food Dataset) is a free
public dataset of 2978 food images covering 19 food types. Each food portion was
photographed as a **top view** and a **side view**, with a **1-Yuan coin
(diameter 25.0 mm)** as the calibration object. Ground-truth **volume (mL)** and
**weight (g)** for every portion are stored in ``density.xls``; per-image object
bounding boxes (food + coin) are stored as PASCAL-VOC XML in ``Annotations/``.

This makes ECUSTFD the public dataset that most closely matches our capture
protocol (two views + a circular reference + measured ground truth), so we use it
to train and evaluate the volume regressor.

Reference:
    Liang & Li, "Computer vision-based food calorie estimation: dataset, method,
    and experiment", arXiv:1705.07632, 2017.
    Repository: https://github.com/Liang-yc/ECUSTFD-resized-

Run::

    python data/download_ecustfd.py
"""
from __future__ import annotations

import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Allow running as a plain script (python data/download_ecustfd.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foodvol import config  # noqa: E402

TARBALL_URL = "https://github.com/Liang-yc/ECUSTFD-resized-/archive/refs/heads/master.tar.gz"
EXPECTED_IMAGES = 2978


def is_downloaded() -> bool:
    images = config.ECUSTFD_DIR / "JPEGImages"
    density = config.ECUSTFD_DIR / "density.xls"
    return density.exists() and images.is_dir() and len(list(images.glob("*.JPG"))) > 1000


def download(force: bool = False) -> Path:
    """Download and extract ECUSTFD into ``data/ECUSTFD``. Idempotent."""
    if is_downloaded() and not force:
        print(f"ECUSTFD already present at {config.ECUSTFD_DIR}")
        return config.ECUSTFD_DIR

    config.ECUSTFD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print(f"Downloading ECUSTFD (~125 MB) from {TARBALL_URL} ...")
        urllib.request.urlretrieve(TARBALL_URL, tmp_path)
        print(f"Extracting into {config.ECUSTFD_DIR} ...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            members = tar.getmembers()
            root_prefix = members[0].name.split("/")[0] + "/"
            for m in members:
                rel = m.name[len(root_prefix):] if m.name.startswith(root_prefix) else m.name
                if not rel:
                    continue
                m.name = rel
                tar.extract(m, config.ECUSTFD_DIR)  # noqa: S202 (trusted GitHub tarball)
    finally:
        tmp_path.unlink(missing_ok=True)

    n = len(list((config.ECUSTFD_DIR / "JPEGImages").glob("*.JPG")))
    print(f"Done: {n} images (expected {EXPECTED_IMAGES}).")
    return config.ECUSTFD_DIR


if __name__ == "__main__":
    force = "--force" in sys.argv
    download(force=force)
