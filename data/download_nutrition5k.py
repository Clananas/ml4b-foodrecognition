"""Download a curated single-ingredient subset of Nutrition5k.

Nutrition5k (Thames et al., CVPR 2021) is Google's open dataset of ~5,000
realistic cafeteria dishes, each with per-ingredient mass, calories, and
macronutrients measured on a scale. The full release is ~181 GB, which is too
much for a laptop. We download only the subset relevant to this project:

    439 single-ingredient standard-food dishes (apple, pizza, broccoli, bacon,
    chicken, rice, …) that have an overhead RGB image. Total size ≈ 220 MB.

Idempotent: re-running skips dishes that are already present.

Reference:
    Thames, Q. et al. "Nutrition5k: Towards Automatic Nutritional Understanding
    of Generic Food." CVPR 2021. https://arxiv.org/abs/2103.03375
    Dataset: https://github.com/google-research-datasets/Nutrition5k
    License: CC BY 4.0

Run:
    python data/download_nutrition5k.py
"""
from __future__ import annotations

import csv
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foodvol import config  # noqa: E402  (sets up SSL_CERT_FILE for downloads)

ROOT = Path(__file__).resolve().parent.parent
META_DIR = ROOT / "data" / "n5k_meta"
IMG_DIR = ROOT / "data" / "Nutrition5k" / "overhead_rgb"
MANIFEST = META_DIR / "n5k_subset_manifest.csv"

BASE_URL = ("https://storage.googleapis.com/nutrition5k_dataset/"
            "nutrition5k_dataset/imagery/realsense_overhead")


def is_downloaded() -> bool:
    if not MANIFEST.exists():
        return False
    expected = sum(1 for _ in open(MANIFEST)) - 1
    actual = sum(1 for _ in IMG_DIR.glob("*.png")) if IMG_DIR.exists() else 0
    return expected > 0 and actual >= int(0.95 * expected)   # tolerate a few missing


def _download_one(dish_id: str, dest: Path, retries: int = 3) -> bool:
    """Fetch the overhead RGB image for ``dish_id``. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    url = f"{BASE_URL}/{dish_id}/rgb.png"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            if len(data) < 1024:           # tiny payload = error page, not a real image
                return False
            dest.write_bytes(data)
            return True
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return False


def download() -> Path:
    """Download every image referenced by the manifest. Idempotent."""
    if not MANIFEST.exists():
        sys.exit(f"ERROR: manifest missing at {MANIFEST}.\n"
                 "       The repository ships with it; if you removed it, "
                 "regenerate from the Nutrition5k metadata.")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    print(f"Nutrition5k subset: {len(rows)} dishes, ~{len(rows) * 0.5:.0f} MB target")
    n_ok = n_fail = n_skip = 0
    for i, row in enumerate(rows, 1):
        did = row["dish_id"]
        dest = IMG_DIR / f"{did}.png"
        if dest.exists() and dest.stat().st_size > 1024:
            n_skip += 1
        elif _download_one(did, dest):
            n_ok += 1
        else:
            n_fail += 1
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  ok={n_ok}  cached={n_skip}  failed={n_fail}")
    total_mb = sum(p.stat().st_size for p in IMG_DIR.glob("*.png")) / (1024 ** 2)
    print(f"\nDone. {IMG_DIR} now contains {n_ok + n_skip}/{len(rows)} images "
          f"({total_mb:.0f} MB).")
    if n_fail:
        print(f"  Note: {n_fail} images could not be downloaded; re-run the script "
              "to retry just those.")
    return IMG_DIR


if __name__ == "__main__":
    download()
