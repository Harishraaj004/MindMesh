"""Vercel entrypoint: exposes the FastAPI app from backend/mindmesh."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from mindmesh.api import app  # noqa: E402,F401
