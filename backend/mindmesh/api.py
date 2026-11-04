import base64
import hmac
import ipaddress
import socket
import threading
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, ingest, reasoning
from .evaluate import run_eval
from .models import embedder, llm
from .retrieval import Index
from .storage import Store
from .visuals import explain_visually

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
store = Store(config.DATABASE_URL, schema=config.DB_SCHEMA or None)
index = Index(store)
app = FastAPI(title="MindMesh")

# Load the embedding model in the background so the first upload doesn't pay the ~15 s load.
if not config.ON_VERCEL and config.EMBEDDER == "minilm":
    threading.Thread(target=lambda: embedder.embed(["warm up"]), daemon=True).start()


@app.middleware("http")
async def password_gate(request: Request, call_next):
    """HTTP Basic gate over the whole app (any username). The browser caches the credentials, so fetch()
    and <img> requests from the UI are covered too."""
    if config.ACCESS_PASSWORD:
        ok = False
        h = request.headers.get("authorization", "")
        if h.lower().startswith("basic "):
            try:
                _, _, pw = base64.b64decode(h[6:]).decode().partition(":")
                ok = hmac.compare_digest(pw.encode(), config.ACCESS_PASSWORD.encode())
            except Exception:
                ok = False
        if not ok:
            return Response("Authentication required", 401, headers={"WWW-Authenticate": 'Basic realm="MindMesh"'})
    return await call_next(request)


def _public_host(url):
    """Reject URLs that resolve to loopback/private/link-local addresses (basic SSRF guard)."""
    host = urlparse(url).hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


class AskReq(BaseModel):
    question: str
    mode: str = "hybrid_graph"
    visual: bool = False


class LinkReq(BaseModel):
    url: str


@app.get("/api/status")
def status():
    return {"llm": llm.name, "llm_configured": llm.available, "embedder": embedder.name,
            "documents": len(store.list_documents())}


@app.get("/api/documents")
def documents():
    return store.list_documents()


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)):
    results, errors = [], []
    for f in files:
        try:
            results.append(ingest.ingest_bytes(store, Path(f.filename).name, await f.read()))
        except Exception as e:
            errors.append({"name": f.filename, "error": str(e)})
    return {"ingested": results, "errors": errors}


@app.post("/api/links")
def add_link(req: LinkReq):
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL must start with http:// or https://")
    if config.ON_VERCEL and not _public_host(req.url):
        raise HTTPException(400, "URL must point to a public host")
    try:
        return ingest.ingest_url(store, req.url)
    except Exception as e:
        raise HTTPException(400, str(e))


def _remove_document(doc):
    if doc["path"]:
        Path(doc["path"]).unlink(missing_ok=True)
    if (doc.get("meta") or {}).get("annotated"):
        Path(doc["meta"]["annotated"]).unlink(missing_ok=True)
    store.delete_document(doc["id"])


@app.delete("/api/documents/{doc_id}")
def delete(doc_id: str):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404)
    _remove_document(doc)
    return {"ok": True}


@app.post("/api/documents/{doc_id}/reindex")
def reindex(doc_id: str):
    """Re-process a stored file with the current settings (new model, new extractor...). The old copy is only
    removed after the new one was indexed successfully."""
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404)
    if not doc["path"] or not Path(doc["path"]).exists():
        raise HTTPException(400, "No stored file for this document (web links must be added again).")
    try:
        res = ingest.ingest_bytes(store, doc["name"], Path(doc["path"]).read_bytes(), source=doc.get("source") or "")
    except Exception as e:
        raise HTTPException(400, str(e))
    _remove_document(doc)
    return res


@app.get("/api/documents/{doc_id}/annotated")
def annotated(doc_id: str):
    doc = store.get_document(doc_id)
    p = ((doc or {}).get("meta") or {}).get("annotated")
    if not p or not Path(p).exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/documents/{doc_id}/file")
def raw_file(doc_id: str):
    doc = store.get_document(doc_id)
    if not doc or not doc["path"] or not Path(doc["path"]).exists():
        raise HTTPException(404)
    return FileResponse(doc["path"], filename=doc["name"])


@app.get("/api/search")
def search(q: str, mode: str = "hybrid_graph", k: int = 8):
    hits, trace = index.search(q, k=k, mode=mode)
    return {"hits": hits, "trace": trace}


@app.post("/api/ask")
def ask(req: AskReq):
    if req.mode not in ("vector", "bm25", "hybrid", "hybrid_graph"):
        raise HTTPException(400, "bad mode")
    res = reasoning.ask(index, req.question, mode=req.mode)
    if (req.visual or res["wants_visual"]) and res["sources"]:
        res["visual"] = explain_visually(req.question, res["answer"], res["bridges"], res["graph_paths"])
    return res


@app.post("/api/samples")
def load_samples():
    have = {d["name"] for d in store.list_documents()}
    loaded = 0
    for p in sorted((config.ROOT / "data" / "samples").glob("*")):
        if p.is_file() and p.name not in have:
            ingest.ingest_bytes(store, p.name, p.read_bytes())
            loaded += 1
    return {"loaded": loaded}


@app.get("/api/graph")
def graph():
    index.refresh()
    g = index.graph.export()
    g["bridges"] = index.graph.cross_doc_bridges()
    return g


@app.post("/api/eval")
def evaluate(modes: list[str] | None = None):
    index.refresh()
    return run_eval(index, modes)


@app.exception_handler(Exception)
async def err(_, exc):
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="ui")
