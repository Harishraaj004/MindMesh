"""Ingestion: parse -> classify -> chunk -> embed -> extract entities -> store."""
import csv
import io
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from . import config
from .graph import extract
from .models import embedder, llm, prep_image, segment_image
from .textutil import chunk_text

CODE_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".cs", ".rb",
            ".php", ".sql", ".sh", ".yaml", ".yml", ".json", ".toml"}
TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".log", ".vtt", ".srt"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


def classify(name):
    ext = Path(name).suffix.lower()
    if ext == ".pdf": return "pdf"
    if ext == ".docx": return "word"
    if ext in (".html", ".htm"): return "web"
    if ext == ".doc": return "legacy_doc"
    if ext in IMAGE_EXT: return "image"
    if ext in CODE_EXT: return "code"
    if ext in (".csv", ".tsv"): return "table"
    if ext in TEXT_EXT or not ext: return "text"
    return "unsupported"


def annotate(png, segs, out_path):
    """Draw SAM boxes + labels on the image so the user can see what was found."""
    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(png)).convert("RGB")
    d, (W, H) = ImageDraw.Draw(im), im.size
    colors = ["#e5333a", "#2f8fe5", "#2fa57a", "#e5a02f", "#b455d6"]
    for i, s in enumerate(segs):
        c = colors[i % len(colors)]
        for x1, y1, x2, y2 in s["boxes"]:
            d.rectangle((x1 * W, y1 * H, x2 * W, y2 * H), outline=c, width=max(2, W // 300))
            d.text((x1 * W + 4, y1 * H + 3), s["label"], fill=c)
    im.save(out_path)


def parse_pdf(data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return [(f"p.{i + 1}", (p.extract_text() or "")) for i, p in enumerate(reader.pages)]


def parse_docx(data):
    """Word (.docx) -> [(section heading, text)]. Headings start sections; tables become 'col: value' rows."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = Document(io.BytesIO(data))
    sections, head, buf = [], "start", []

    def flush():
        if any(b.strip() for b in buf):
            sections.append((head, "\n\n".join(b for b in buf if b.strip())))

    for el in doc.element.body.iterchildren():
        if el.tag.endswith("}p"):
            p = Paragraph(el, doc)
            text = p.text.strip()
            if not text:
                continue
            if (p.style.name or "").lower().startswith(("heading", "title")):
                flush(); buf.clear(); head = text[:80]
                buf.append(text)
            else:
                buf.append(text)
        elif el.tag.endswith("}tbl"):
            rows = [[c.text.strip() for c in r.cells] for r in Table(el, doc).rows]
            if rows:
                hdr = rows[0]
                buf.append("Table: " + "\n".join(", ".join(f"{h}: {v}" for h, v in zip(hdr, r)) for r in rows[1:])
                           if len(rows) > 1 else "Table: " + ", ".join(hdr))
    flush()
    return sections


def parse_table(data, name):
    text = data.decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text), delimiter="\t" if name.endswith(".tsv") else ","))
    if not rows:
        return []
    head, body = rows[0], rows[1:]
    out, step = [], 15
    for i in range(0, len(body), step):  # serialize rows as "col: value" so they are searchable
        lines = [", ".join(f"{h}: {v}" for h, v in zip(head, r)) for r in body[i:i + step]]
        out.append((f"rows {i + 1}-{i + len(lines)}", f"Table columns: {', '.join(head)}.\n" + "\n".join(lines)))
    return out


def parse_code(text):
    """Split on top-level def/class boundaries so each chunk is one logical unit."""
    parts = re.split(r"\n(?=(?:def |class |function |export |public |async def ))", text)
    return [(f"unit {i + 1}", p) for i, p in enumerate(parts) if p.strip()]


def html_to_text(html):
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    title = re.search(r"(?is)<title>(.*?)</title>", html)
    html = re.sub(r"(?s)<[^>]+>", "\n", html)
    html = re.sub(r"&nbsp;", " ", html)
    text = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", html))
    return (title.group(1).strip() if title else ""), text


def _index(store, doc_id, units, kind, modality="text"):
    chunks = []
    for loc, text in units:
        if kind in ("code", "table"):
            pieces = [text] if len(text.split()) <= config.CHUNK_WORDS * 2 else chunk_text(text, config.CHUNK_WORDS * 2, 20)
        else:
            pieces = chunk_text(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP)
        chunks += [{"text": p, "modality": modality if kind != "code" else "code", "loc": loc} for p in pieces]
    if not chunks:
        return 0
    embs = embedder.embed([c["text"] for c in chunks])
    ids = store.add_chunks(doc_id, chunks, embs)
    # Model calls are network-bound, so run them in parallel; DB writes stay on this thread, in order.
    workers = config.EXTRACT_WORKERS if llm.available and config.ENTITY_EXTRACTION == "llm" else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda c: extract(c["text"], is_code=(kind == "code")), chunks)
        for cid, (ents, rels) in zip(ids, results):
            store.add_entities(cid, ents, rels)
    return len(chunks)


def ingest_bytes(store, name, data, source=""):
    kind = classify(name)
    if kind == "legacy_doc":
        raise ValueError(f"{name}: legacy .doc is not supported. Open it in Word and save as .docx.")
    if kind == "unsupported":
        raise ValueError(f"Unsupported file type: {name}")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"bytes": len(data)}
    doc_id = store.add_document(name, kind, source=source, meta=meta)
    path = config.UPLOAD_DIR / f"{doc_id}{Path(name).suffix.lower()}"
    path.write_bytes(data)
    store.set_path(doc_id, str(path))

    if kind == "image":
        png, mime = prep_image(data)  # PNG, EXIF-rotated, downscaled: accepted by every model
        desc, objects, note = None, [], None
        try:
            res = llm.describe_image(png, mime)  # Muse Spark vision: description + OCR + object list
            if res:
                desc, objects = res
        except Exception as e:  # an analysis failure must not lose the upload
            note = f"Image analysis failed ({type(e).__name__}); indexed by filename only."
        segs = segment_image(png, mime, objects)  # SAM 3.1 locates each named object
        parts = [f"Image file: {name}."]
        if desc:
            parts.append(desc)
        if segs:
            parts.append("Objects located by SAM 3.1: " + "; ".join(f"{s['label']} x{s['count']}" for s in segs) + ".")
        if not desc:
            parts.append(note or "No vision model configured; indexed by filename only. "
                                 "Set MINDMESH_LLM_MODEL (or MINDMESH_VISION_MODEL) to analyze images.")
        meta.update(captioned=bool(desc), objects=objects,
                    segments=[{"label": s["label"], "count": s["count"], "boxes": [list(b) for b in s["boxes"]]}
                              for s in segs])
        if segs:
            annotated = config.UPLOAD_DIR / f"{doc_id}_annotated.png"
            annotate(png, segs, annotated)
            meta["annotated"] = str(annotated)
        store.set_meta(doc_id, meta)
        n = _index(store, doc_id, [("image", "\n".join(parts))], kind, modality="image")
    else:
        if kind == "pdf": units = parse_pdf(data)
        elif kind == "word": units = parse_docx(data)
        elif kind == "web": units = [("page", html_to_text(data.decode("utf-8", errors="replace"))[1])]
        elif kind == "table": units = parse_table(data, name)
        else:
            text = data.decode("utf-8", errors="replace")
            units = parse_code(text) if kind == "code" else [("text", text)]
        n = _index(store, doc_id, units, kind)
    if n == 0:
        store.delete_document(doc_id)
        raise ValueError(f"No extractable text in {name} (scanned PDF? OCR is not enabled).")
    return {"id": doc_id, "name": name, "kind": kind, "chunks": n}


def ingest_url(store, url):
    r = httpx.get(url, follow_redirects=True, timeout=30, headers={"User-Agent": config.USER_AGENT})
    if r.status_code in (401, 403, 429):
        hint = (" Wikimedia sites require contact info in the User-Agent: set MINDMESH_USER_AGENT in .env "
                "(see .env.example)." if "wikipedia.org" in url or "wikimedia.org" in url else "")
        raise ValueError(f"The site refused the request ({r.status_code}).{hint} "
                         "Alternatively save the page as HTML/PDF and upload the file.")
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype:
        return ingest_bytes(store, Path(url).name or "download.pdf", r.content, source=url)
    title, text = html_to_text(r.text)
    name = (title or url)[:80]
    doc_id = store.add_document(name, "web", source=url)
    n = _index(store, doc_id, [("page", text)], "web")
    if n == 0:
        store.delete_document(doc_id)
        raise ValueError("No extractable text at that URL.")
    return {"id": doc_id, "name": name, "kind": "web", "chunks": n}
