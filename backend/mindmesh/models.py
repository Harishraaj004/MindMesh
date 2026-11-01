"""Model layer, built on LangChain chat/embedding interfaces.

Swap providers with MINDMESH_PROVIDER (no other code changes):
  meta       (default) any OpenAI-compatible endpoint: Meta's Muse Spark API, or Ollama/vLLM open-weight
  openai     OpenAI proper (ignores MINDMESH_LLM_BASE_URL)
  anthropic  Claude (needs `pip install langchain-anthropic`)

When no model is configured, `llm.available` is False and callers use the deterministic local fallback.
SAM and Muse Image are not chat models, so they stay as small HTTP adapters below.
"""
import base64
import io
import json
import re
import zlib
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .textutil import tokens

DIM = 384  # all-MiniLM-L6-v2; must match storage.EMBED_DIM


# ------------------------------------------------------------------ chat models
def make_chat_model(model=None, effort=None):
    """Return a LangChain chat model for the configured provider, or None if unconfigured."""
    provider = config.PROVIDER
    model = model or config.LLM_MODEL
    if not model:
        return None
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError("MINDMESH_PROVIDER=anthropic needs: pip install langchain-anthropic") from e
        return ChatAnthropic(model=model, api_key=config.LLM_API_KEY, temperature=0.2, max_tokens=4000)
    from langchain_openai import ChatOpenAI
    if provider == "meta":
        if not config.LLM_BASE_URL:
            return None
        extra = {"reasoning_effort": effort} if effort else {}  # Muse Spark is a reasoning model
        return ChatOpenAI(model=model, base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY or "none",
                          temperature=0.2, max_tokens=4000, timeout=180, **extra)
    if provider == "openai":
        return ChatOpenAI(model=model, api_key=config.LLM_API_KEY, temperature=0.2, max_tokens=4000, timeout=180)
    raise ValueError(f"Unknown MINDMESH_PROVIDER: {provider}")


class LLM:
    """Thin facade over the LangChain chat model so the rest of the code stays provider-agnostic."""

    def __init__(self):
        self.model = make_chat_model(effort=config.REASONING_EFFORT)
        self.fast = make_chat_model(effort=config.FAST_EFFORT)  # planning + entity extraction
        self.vision = make_chat_model(config.VISION_MODEL, effort=config.FAST_EFFORT) if config.VISION_MODEL else None

    @property
    def available(self):
        return self.model is not None

    @property
    def name(self):
        return f"{config.PROVIDER}:{config.LLM_MODEL}" if self.available else "local-extractive (no LLM configured)"

    @staticmethod
    def _msgs(messages):
        cls = {"system": SystemMessage, "user": HumanMessage}
        return [cls[m["role"]](content=m["content"]) for m in messages]

    def chat(self, messages, fast=False):
        out = ((self.fast or self.model) if fast else self.model).invoke(self._msgs(messages)).content
        if isinstance(out, list):  # some providers return content blocks
            out = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in out)
        if not out or not out.strip():  # e.g. a reasoning model spent its whole token budget thinking
            raise ValueError("model returned an empty response")
        return out

    def json(self, system, user):
        """Provider-neutral JSON: ask for JSON, then parse the first {...} block."""
        txt = self.chat([{"role": "system", "content": system + " Respond with JSON only."},
                         {"role": "user", "content": user}], fast=True)
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0) if m else txt)

    def describe_image(self, data: bytes, mime: str):
        """Describe an image with a multimodal chat model -> (description, [object phrases]).
        None when no vision model is configured."""
        if not self.vision:
            return None
        content = [{"type": "text", "text": (
            "Analyze this image for a search index. Respond with JSON only: "
            '{"description": "dense factual description; transcribe ALL visible text verbatim; say what kind of '
            'image it is (logo, photo, chart, screenshot, diagram...)", '
            '"objects": ["up to 5 short noun phrases naming distinct visual elements, e.g. \'company logo\', '
            '\'building\', \'bar chart\'"]}')},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}]
        out = self.vision.invoke([HumanMessage(content=content)]).content
        if isinstance(out, list):
            out = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in out)
        if not out or not out.strip():
            raise ValueError("vision model returned an empty response")
        m = re.search(r"\{.*\}", out, re.S)
        try:
            j = json.loads(m.group(0)) if m else {}
            objs = [o for o in j.get("objects", []) if isinstance(o, str) and o.strip()][:5]
            return str(j.get("description") or out).strip(), objs
        except (ValueError, AttributeError):
            return out.strip(), []


# ------------------------------------------------------------------ embeddings
class HashingEmbeddings(Embeddings):
    """Signed feature hashing over unigrams + bigrams. Offline/deterministic fallback; lexical, NOT neural.
    Same 384 dims as MiniLM so it fits the pgvector column. Used by tests (MINDMESH_EMBEDDER=hashing)."""

    def embed_documents(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokens(t)
            feats = [(w, 1.0) for w in toks] + [(a + "_" + b, 0.6) for a, b in zip(toks, toks[1:])]
            for f, w in feats:
                h = zlib.crc32(f.encode())
                out[i, h % DIM] += w * (1 if (h >> 20) & 1 else -1)
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out.tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class Embedder:
    """numpy-friendly wrapper over a LangChain Embeddings object.

    Default: sentence-transformers/all-MiniLM-L6-v2 (384 dims, runs locally; weights download from the
    Hugging Face Hub on first use and are cached). The model loads lazily on the first embed call.
    """

    def __init__(self):
        self._lc = None
        self.name = "hashing-384" if config.EMBEDDER == "hashing" else config.EMBED_MODEL_NAME

    @property
    def lc(self):
        if self._lc is None:
            if config.EMBEDDER == "hashing":
                self._lc = HashingEmbeddings()
            else:
                from langchain_huggingface import HuggingFaceEmbeddings
                self._lc = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL_NAME,
                                                 encode_kwargs={"normalize_embeddings": True})
        return self._lc

    def embed(self, texts):
        m = np.asarray(self.lc.embed_documents(list(texts)), dtype=np.float32)
        if m.shape[1] != DIM:
            raise ValueError(f"Embedding model returned {m.shape[1]} dims; the database expects {DIM}.")
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ Meta image models (same API and key as chat)
SAM_BOX_RE = re.compile(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")


def _meta_ready(model):
    return bool(model and config.PROVIDER == "meta" and config.LLM_BASE_URL and config.LLM_API_KEY)


def _meta_headers():
    return {"Authorization": f"Bearer {config.LLM_API_KEY}"}


def prep_image(data: bytes, max_side=1536):
    """Normalize any uploaded image to PNG (EXIF-rotated, downscaled) so every model call gets a supported format."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue(), "image/png"


def parse_sam_boxes(text):
    """SAM 3.1 answers with text tokens, one <|box;x1;y1;x2;y2;w;h|> per instance (empty text = no match).
    Returns boxes as fractions of the image size: [(x1, y1, x2, y2), ...]."""
    out = []
    for x1, y1, x2, y2, w, h in SAM_BOX_RE.findall(text or ""):
        w, h = int(w) or 1, int(h) or 1
        out.append((int(x1) / w, int(y1) / h, int(x2) / w, int(y2) / h))
    return out


def segment_image(data: bytes, mime: str, concepts):
    """Locate each text concept in the image with SAM 3.1 (Meta Responses API).
    Returns [{"label", "count", "boxes": [(x1,y1,x2,y2) as fractions]}] for concepts that were found."""
    if not _meta_ready(config.SAM_MODEL) or not concepts:
        return []
    uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"

    def one(concept):
        try:
            r = httpx.post(config.LLM_BASE_URL + "/responses", headers=_meta_headers(), timeout=120, json={
                "model": config.SAM_MODEL,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": concept},
                                                       {"type": "input_image", "image_url": uri}]}]})
            r.raise_for_status()
            text = "".join(c.get("text", "") for o in r.json().get("output", []) for c in o.get("content", []))
            boxes = parse_sam_boxes(text)
            return {"label": concept, "count": len(boxes), "boxes": boxes} if boxes else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=5) as pool:
        return [r for r in pool.map(one, concepts) if r]


def generate_image(prompt: str):
    """Muse Image (Meta /images/generations). Returns {"mime", "data": base64} or None when unavailable/failing."""
    if not _meta_ready(config.IMAGE_MODEL):
        return None
    try:
        r = httpx.post(config.LLM_BASE_URL + "/images/generations", headers=_meta_headers(), timeout=180,
                       json={"model": config.IMAGE_MODEL, "prompt": prompt[:3500], "n": 1})
        r.raise_for_status()
        b64 = r.json()["data"][0]["b64_json"]
        head = base64.b64decode(b64[:32])
        mime = "image/webp" if head[:4] == b"RIFF" else "image/jpeg" if head[:2] == b"\xff\xd8" else "image/png"
        return {"mime": mime, "data": b64}
    except Exception:
        return None


llm = LLM()
embedder = Embedder()
