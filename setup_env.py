"""Cross-platform environment setup for the food-volume project.

Creates a local virtual environment, installs all dependencies, installs the
``foodvol`` package in editable mode and registers a Jupyter kernel. Works on
Windows, Linux and macOS.

Usage (run with a Python 3.10-3.12 interpreter)::

    python setup_env.py

Then activate the environment:

    Windows (PowerShell):   .venv\\Scripts\\Activate.ps1
    Windows (cmd):          .venv\\Scripts\\activate.bat
    Linux / macOS:          source .venv/bin/activate
"""
from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
KERNEL_NAME = "ml4b-foodvol"
KERNEL_DISPLAY = "Python (ml4b-foodvol)"


def venv_python(venv_dir: Path) -> Path:
    """Path to the venv's Python interpreter (platform-dependent)."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd)


def check_python_version() -> None:
    major, minor = sys.version_info[:2]
    print(f"Using Python {major}.{minor} ({sys.executable})")
    if (major, minor) < (3, 10):
        sys.exit("ERROR: Python 3.10 or newer is required.")
    if (major, minor) >= (3, 13):
        print(
            "WARNING: Python 3.13+ is very new — some ML wheels (PyTorch etc.) may not\n"
            "         be available yet. If installation fails, use Python 3.10-3.12."
        )


def main() -> None:
    check_python_version()

    if VENV_DIR.exists():
        print(f"Virtual environment already exists at {VENV_DIR} (reusing).")
    else:
        print(f"Creating virtual environment at {VENV_DIR} ...")
        venv.EnvBuilder(with_pip=True, upgrade_deps=False).create(VENV_DIR)

    py = venv_python(VENV_DIR)
    if not py.exists():
        sys.exit(f"ERROR: expected venv interpreter not found at {py}")

    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    run([str(py), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")])
    run([str(py), "-m", "pip", "install", "-e", str(PROJECT_ROOT)])
    run([str(py), "-m", "ipykernel", "install", "--user",
         "--name", KERNEL_NAME, "--display-name", KERNEL_DISPLAY])

    activate = (".venv\\Scripts\\Activate.ps1" if sys.platform == "win32"
                else "source .venv/bin/activate")
    print(
        "\n" + "=" * 64 +
        "\nSetup complete.\n"
        f"  1. Activate the environment:   {activate}\n"
        "  2. Download the dataset:       python data/download_ecustfd.py\n"
        "  3. Run the app:                streamlit run app.py\n"
        "  4. Open the notebook:          jupyter lab notebooks/00_feasibility.ipynb\n"
        + "=" * 64
    )


if __name__ == "__main__":
    main()
