"""Runtime configuration, read from environment variables (see .env.example)."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Minimal .env loader (KEY=VALUE lines; real environment variables win).
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"\''))
DATA_DIR = Path(os.getenv("MINDMESH_DATA_DIR", ROOT / "data" / "store"))
UPLOAD_DIR = DATA_DIR / "uploads"
FRONTEND_DIR = ROOT / "frontend"

# meta (default) | openai | anthropic. Meta = any OpenAI-compatible endpoint (see models.py).
PROVIDER = os.getenv("MINDMESH_PROVIDER", "meta").lower()

# Any OpenAI-compatible chat endpoint: a Meta-hosted Muse Spark deployment,
# or an open-weight model served by Ollama / vLLM / llama.cpp.
LLM_BASE_URL = re.sub(r"/(responses|chat/completions)/?$", "", os.getenv("MINDMESH_LLM_BASE_URL", "").strip().rstrip("/"))
LLM_API_KEY = os.getenv("MINDMESH_LLM_API_KEY", "")
LLM_MODEL = os.getenv("MINDMESH_LLM_MODEL", "")
# Multimodal chat model used to describe uploaded images. Defaults to the main model (Muse Spark reads images).
# Do NOT set this to muse-image-1.0: that model generates images, it does not describe them.
VISION_MODEL = os.getenv("MINDMESH_VISION_MODEL") or LLM_MODEL

# Reasoning effort for Muse Spark (meta provider): answers use REASONING_EFFORT, planning and entity extraction
# use the cheaper FAST_EFFORT. Values: minimal | low | medium | high. Higher = slower and uses more tokens.
REASONING_EFFORT = os.getenv("MINDMESH_REASONING_EFFORT") or "low"
FAST_EFFORT = os.getenv("MINDMESH_FAST_EFFORT") or "minimal"

# Embeddings: local sentence-transformers model (384 dims). MINDMESH_EMBEDDER=hashing gives an offline lexical fallback.
EMBEDDER = os.getenv("MINDMESH_EMBEDDER", "minilm").lower()
EMBED_MODEL_NAME = os.getenv("MINDMESH_EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"

# PostgreSQL with the pgvector extension. The database is created on first start if it does not exist.
DATABASE_URL = os.getenv("MINDMESH_DATABASE_URL", "")
DB_SCHEMA = os.getenv("MINDMESH_DB_SCHEMA", "")  # optional isolated schema (tests)

# Meta image models, called on the same API/key as the chat model (provider "meta" only). Blank = disabled.
SAM_MODEL = os.getenv("MINDMESH_SAM_MODEL", "")      # e.g. sam-3.1: locates objects in uploaded images
IMAGE_MODEL = os.getenv("MINDMESH_IMAGE_MODEL", "")  # e.g. muse-image-1.0: draws visual explanations

# Entity extraction: "llm" (typed relations, one model call per chunk, run in parallel) or "heuristic" (instant, noisier).
ENTITY_EXTRACTION = (os.getenv("MINDMESH_ENTITY_EXTRACTION") or "llm").lower()
EXTRACT_WORKERS = int(os.getenv("MINDMESH_EXTRACT_WORKERS") or "8")

# Sent when fetching web links. Wikipedia/Wikimedia require contact info here, e.g.
# MINDMESH_USER_AGENT="MindMesh/0.1 (https://your-site.example; you@example.com)"
USER_AGENT = os.getenv("MINDMESH_USER_AGENT") or "MindMesh/0.1 (local research prototype)"

CHUNK_WORDS = int(os.getenv("MINDMESH_CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("MINDMESH_CHUNK_OVERLAP", "30"))
