"""Checks the real sentence-transformers/all-MiniLM-L6-v2 model (downloads ~90 MB on first run)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_minilm_is_384d_and_semantic():
    from mindmesh import config
    from mindmesh.models import Embedder
    config.EMBEDDER = "minilm"
    e = Embedder()
    v = e.embed(["The sensor reading drifts upward over time.",
                 "Measurements slowly creep higher as the probe ages.",
                 "The recipe calls for two cups of flour."])
    assert v.shape == (3, 384) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-3)
    assert v[0] @ v[1] > v[0] @ v[2] + 0.2  # paraphrase is much closer than an unrelated sentence
