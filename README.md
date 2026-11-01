# MindMesh

**A multimodal AI knowledge workspace.** Upload PDFs, Word files, images, code, spreadsheets and web pages, ask questions across all of them, and get answers that cite the exact passages they came from, along with a knowledge graph of how your sources connect.

> Personal learning project exploring retrieval-augmented generation (RAG), hybrid search, knowledge graphs and agentic LLM workflows.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1C3C3C)

---

## Features

- **Mixed inputs:** PDF, Word (`.docx`), plain text and Markdown, source code, CSV/TSV, images and web links.
- **Hybrid retrieval:** vector search (pgvector) and BM25 keyword search, merged with reciprocal rank fusion, then expanded through the knowledge graph so related passages from *other* documents are pulled in.
- **Cited answers:** every statement points to a source passage (`[S1]`, `[S2]`, …). Click a citation to open the passage.
- **Self-checking:** a LangGraph pipeline verifies that each citation exists and that its text supports the statement, and rewrites the answer once if it doesn't.
- **Knowledge graph:** entities and relations are extracted from every chunk. Entities that appear in more than one document ("bridges") show how separate sources connect. Explore them in an interactive graph view.
- **Image understanding:** with the Meta provider, uploaded images are described by a vision model and objects are located with SAM 3.1, then shown with bounding boxes.
- **Visual explanations:** optionally draws a diagram of the answer (AI-generated image, or a built-in SVG concept map).
- **Evaluation harness:** compares 4 retrieval modes on 6 metrics from the UI or the API.
- **Works offline:** with no LLM configured it still runs, using local embeddings and an extractive answerer that quotes sources and abstains when the evidence is thin.
- **Switchable providers:** Meta (any OpenAI-compatible endpoint, including Ollama and vLLM), OpenAI or Anthropic Claude, chosen in `.env`.

## How it works

```mermaid
flowchart LR
    A[Upload files / links] --> B[Ingest & chunk]
    B --> C[Embeddings<br/>MiniLM, 384-d]
    B --> D[Entity & relation<br/>extraction]
    C --> E[(PostgreSQL + pgvector)]
    D --> E
    Q[Question] --> P[Plan<br/>sub-queries]
    P --> R[Retrieve<br/>vector + BM25 → RRF]
    E --> R
    R --> G[Graph expansion<br/>cross-document links]
    G --> GEN[Generate<br/>cited answer]
    GEN --> V{Verify<br/>citations}
    V -- unsupported --> GEN
    V -- ok --> OUT[Answer + sources + graph]
```

The reasoning pipeline is a LangGraph state graph:

```
plan -> retrieve -> context -> generate -> verify --(unsupported citations, LLM connected)--> generate (once) -> END
```

## Tech stack

| Layer | Tools |
|---|---|
| API | Python, FastAPI, Pydantic, Uvicorn |
| Storage | PostgreSQL with pgvector (HNSW cosine index) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384 dims) |
| Orchestration | LangGraph, LangChain |
| LLMs | Meta Muse Spark / any OpenAI-compatible model, OpenAI, Anthropic Claude |
| Vision | Muse Spark (image description), SAM 3.1 (object location) |
| Frontend | Plain HTML, CSS and JavaScript (no build step); canvas-based force-directed graph |
| Testing | pytest |

## Project structure

```
MindMesh/
├── backend/
│   ├── mindmesh/
│   │   ├── api.py          # FastAPI routes; also serves the frontend
│   │   ├── config.py       # settings read from .env
│   │   ├── ingest.py       # file / URL parsing and chunking
│   │   ├── models.py       # LLM, embedding and vision adapters
│   │   ├── storage.py      # PostgreSQL + pgvector persistence
│   │   ├── graph.py        # knowledge graph and cross-document bridges
│   │   ├── retrieval.py    # vector + BM25 -> RRF -> graph expansion
│   │   ├── reasoning.py    # LangGraph plan/retrieve/generate/verify pipeline
│   │   ├── visuals.py      # visual explanations
│   │   ├── evaluate.py     # evaluation metrics
│   │   └── textutil.py
│   ├── tests/              # end-to-end and embedding tests
│   └── requirements.txt
├── frontend/index.html     # single-page UI
├── data/samples/           # small connected demo corpus
├── eval/dataset.json       # labelled evaluation questions
├── docker-compose.yml      # optional Postgres + pgvector
├── .env.example
└── run.ps1                 # Windows start script
```

## Getting started

### Prerequisites

- Python 3.10 or newer (developed on 3.14)
- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension, **or** Docker
- Optional: an API key for Meta, OpenAI or Anthropic. Without one the app runs in offline mode.

### 1. Clone and install

```bash
git clone https://github.com/Harishraaj004/MindMesh.git
cd MindMesh
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

The first run downloads the MiniLM embedding model (about 90 MB, cached afterwards).

### 2. Start the database

The easiest way is Docker:

```bash
docker compose up -d
```

This starts Postgres with pgvector on port **5440**. You can also use a local PostgreSQL install with the `vector` extension available. The `mindmesh` database is created automatically on first start.

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and set at least the database URL:

```ini
# Docker setup from step 2:
MINDMESH_DATABASE_URL=postgresql://postgres:postgres@localhost:5440/mindmesh
```

To use an LLM, also set the provider, key and model (see [Configuration](#configuration)). Leave them blank to run offline.

### 4. Run

```bash
# Windows
.\run.ps1

# macOS / Linux
uvicorn mindmesh.api:app --app-dir backend --port 8000
```

Open **http://127.0.0.1:8000**.

### 5. Try it

Click **Load samples** to index a small connected corpus (research notes, a meeting transcript, a vendor review, sensor readings and calibration code), then ask:

> *Could the Riverside Station spike be a sensor fault, and what is being done to check?*

The answer draws on several documents. Click a citation to see its passage, or open the **Knowledge graph** tab to see the entities that link them.

## Configuration

All settings are environment variables, read from `.env`. See [`.env.example`](.env.example) for the full list.

| Variable | Purpose | Default |
|---|---|---|
| `MINDMESH_DATABASE_URL` | PostgreSQL connection string (**required**) | – |
| `MINDMESH_PROVIDER` | `meta`, `openai` or `anthropic` | `meta` |
| `MINDMESH_LLM_BASE_URL` | Base URL for OpenAI-compatible endpoints, e.g. `https://api.meta.ai/v1` or Ollama's `http://localhost:11434/v1` (a base URL, not `/v1/responses`) | – |
| `MINDMESH_LLM_API_KEY` | API key for the chosen provider | – |
| `MINDMESH_LLM_MODEL` | Chat model id | – |
| `MINDMESH_VISION_MODEL` | Model that describes uploaded images | the chat model |
| `MINDMESH_SAM_MODEL` | Object location in images, e.g. `sam-3.1` (Meta only) | disabled |
| `MINDMESH_IMAGE_MODEL` | Draws visual explanations, e.g. `muse-image-1.0` (Meta only) | disabled (SVG map) |
| `MINDMESH_EMBEDDER` | `minilm`, or `hashing` for an offline lexical fallback | `minilm` |
| `MINDMESH_ENTITY_EXTRACTION` | `llm` (one call per chunk, more accurate) or `heuristic` (instant, noisier) | `llm` |
| `MINDMESH_EXTRACT_WORKERS` | Parallel extraction calls | `8` |
| `MINDMESH_USER_AGENT` | Sent when fetching web links (Wikipedia requires contact details) | generic |

**Providers**

- **Meta (default):** any OpenAI-compatible endpoint. Also works with open-weight models served by Ollama, vLLM or llama.cpp.
- **OpenAI:** set the key and model; the base URL is ignored.
- **Anthropic:** run `pip install langchain-anthropic`, then set the key and a Claude model id.

> For large uploads, set `MINDMESH_ENTITY_EXTRACTION=heuristic` to avoid one LLM call per chunk.

## API

The UI is a thin client over a REST API. Interactive docs are at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Active models and document count |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/documents` | Upload one or more files (multipart) |
| `POST` | `/api/links` | Add a web page by URL |
| `DELETE` | `/api/documents/{id}` | Remove a document |
| `POST` | `/api/documents/{id}/reindex` | Re-process a file with current settings |
| `POST` | `/api/ask` | Ask a question; returns answer, sources, citations, verification and trace |
| `GET` | `/api/search` | Raw retrieval results for a query |
| `GET` | `/api/graph` | Knowledge graph nodes, links and bridges |
| `POST` | `/api/samples` | Load the demo corpus |
| `POST` | `/api/eval` | Run the evaluation |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Why do Halden Sensor readings drift?", "mode": "hybrid_graph"}'
```

`mode` is one of `vector`, `bm25`, `hybrid` or `hybrid_graph`.

## Evaluation

The **Evaluation** tab (or `POST /api/eval`) runs [`eval/dataset.json`](eval/dataset.json) against all four retrieval modes and reports:

| Metric | Measures |
|---|---|
| Recall@k | Was an expected document retrieved in the top k? |
| MRR | How early the first relevant passage appears |
| Answer accuracy | Share of expected facts present in the answer |
| Citation validity | Do cited ids exist in the retrieved context? |
| Citation support | Does the cited text support the statement? (lexical check) |
| Abstention | Does it decline questions the corpus can't answer? |

> **Note:** the bundled dataset is 10 questions over 5 files, so recall saturates and the modes don't separate much. It is a test harness, not a benchmark. For meaningful numbers, build a larger labelled set from a real corpus.

## Running tests

Tests need a PostgreSQL database (`MINDMESH_DATABASE_URL`). They run in an isolated schema that is dropped afterwards, use offline hashing embeddings, and are skipped if no database is configured.

```bash
python -m pytest backend/tests -q -s
```

## Known limitations

- Word support is `.docx` only; legacy `.doc` is rejected and embedded images are ignored.
- No OCR (scanned PDFs are rejected) and no audio transcription.
- No authentication or access control, and URL ingestion fetches any URL you give it. **Run it locally only.**
- Heuristic entity extraction is noisy without an LLM.
- Generated images can contain incorrect text; always check the cited sources.

## What I learned

- Combining dense and sparse retrieval with reciprocal rank fusion, and when graph expansion helps
- Building agentic LLM workflows as explicit state graphs with LangGraph
- Grounding and verifying LLM output with citations instead of trusting it
- Storing and searching vectors inside PostgreSQL with pgvector
- Designing an evaluation harness and understanding its limits

## Author

**Harish Raaj** · [GitHub @Harishraaj004](https://github.com/Harishraaj004)
