import io, os, sys, tempfile, uuid
from pathlib import Path
os.environ["MINDMESH_DATA_DIR"] = tempfile.mkdtemp()
os.environ["MINDMESH_EMBEDDER"] = "hashing"  # hermetic: no model download; MiniLM is covered by test_minilm.py
os.environ["MINDMESH_DB_SCHEMA"] = "mm_test_" + uuid.uuid4().hex[:8]  # isolated schema, dropped afterwards
for _k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "VISION_MODEL", "EMBED_BASE_URL", "EMBED_MODEL", "SAM_MODEL", "IMAGE_MODEL"):
    os.environ["MINDMESH_" + _k] = ""  # tests must be hermetic: never use the developer's real .env models
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient
from mindmesh import config

if not config.DATABASE_URL:
    pytest.skip("MINDMESH_DATABASE_URL not set (Postgres + pgvector required)", allow_module_level=True)

from mindmesh.api import app, store  # noqa: E402

c = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _drop_schema():
    yield
    store.drop_schema()


def _docx_bytes():
    from docx import Document
    d = Document()
    d.add_heading("Aurora Budget", 1)
    d.add_paragraph("The Zephyr Calibration Fund covers Brightwater Probe trials at Riverside Station.")
    t = d.add_table(rows=2, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "item", "cost"
    t.rows[1].cells[0].text, t.rows[1].cells[1].text = "wiper kit", "310 dollars"
    buf = io.BytesIO(); d.save(buf)
    return buf.getvalue()


def test_word_upload_is_searchable_and_cited():
    c.post("/api/samples")
    r = c.post("/api/documents", files=[("files", ("budget.docx", _docx_bytes(), "application/octet-stream"))]).json()
    assert r["ingested"][0]["kind"] == "word" and not r["errors"]
    res = c.post("/api/ask", json={"question": "What does the Zephyr Calibration Fund cover?"}).json()
    assert any(s["doc"] == "budget.docx" for s in res["sources"])
    tbl = c.post("/api/ask", json={"question": "How much is the wiper kit?"}).json()
    assert "310" in tbl["answer"]
    bad = c.post("/api/documents", files=[("files", ("old.doc", b"x", "application/msword"))]).json()
    assert "legacy .doc" in bad["errors"][0]["error"]


def test_flow():
    c.post("/api/samples")
    assert len(c.get("/api/documents").json()) >= 5
    q = c.post("/api/ask", json={"question": "Why does Halden Sensor drift and what is the correction?", "visual": True}).json()
    assert q["sources"] and q["verification"]["invalid_citations"] == []
    assert len({s["doc"] for s in q["sources"]}) >= 2
    assert q["visual"] is None or q["visual"]["kind"] in ("svg", "image")
    assert c.get("/api/graph").json()["bridges"]
    miss = c.post("/api/ask", json={"question": "warranty on the Quasar X9 spectrometer"}).json()
    assert "couldn't find evidence" in miss["answer"]
    ev = c.post("/api/eval").json()
    for mode, r in ev["results"].items():
        print(mode, r["summary"])


def test_llm_path_revises_bad_citations():
    """A connected model that first cites a bogus source must be sent back once and then pass verification."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from mindmesh import reasoning
    from mindmesh.api import index
    from mindmesh.models import llm

    c.post("/api/samples")
    good = "Halden Sensor turbidity readings drift upward by roughly 4 percent per month when the optical window fouls with algae. [S1]"
    saved = llm.model
    llm.model = FakeListChatModel(responses=['{"queries": ["Halden Sensor turbidity drift per month"]}', "Halden drift is huge. [S9]", good])
    reasoning._compiled.clear()
    try:
        res = reasoning.ask(index, "How much does Halden Sensor turbidity drift each month?")
    finally:
        llm.model = saved
        reasoning._compiled.clear()
    gens = [s for s in res["steps"] if s["step"] == "generate"]
    assert len(gens) == 2 and "revision 1" in gens[1]["detail"]
    assert res["verification"]["invalid_citations"] == []


def test_sam_box_parsing():
    from mindmesh.models import parse_sam_boxes
    txt = "<0f>0<|box;x1=180;y1=50;x2=280;y2=150;w=320;h=200|><|mask;x=0;y=0;data=abc|>,1<|box;x1=20;y1=40;x2=120;y2=140;w=320;h=200|><|mask;x=0;y=0;data=def|>"
    boxes = parse_sam_boxes(txt)
    assert len(boxes) == 2 and boxes[0] == (180 / 320, 50 / 200, 280 / 320, 150 / 200)
    assert parse_sam_boxes("") == []  # SAM returns empty text when nothing matches


def test_reindex_replaces_document_in_place():
    c.post("/api/samples")
    before = c.get("/api/documents").json()
    doc = next(d for d in before if d["name"] == "vendor_review.md")
    r = c.post(f"/api/documents/{doc['id']}/reindex")
    assert r.status_code == 200 and r.json()["name"] == "vendor_review.md"
    after = c.get("/api/documents").json()
    assert len(after) == len(before)  # replaced, not duplicated
    assert doc["id"] not in {d["id"] for d in after}
