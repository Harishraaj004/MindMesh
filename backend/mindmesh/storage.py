"""PostgreSQL + pgvector store: documents, chunk embeddings (vector(384)), entities and relations.

Vector search runs inside Postgres (HNSW, cosine distance). BM25 and the graph are built in memory
from the rows returned here (see retrieval.py).
"""
import json
import threading
import time
import uuid

import numpy as np
import psycopg
import psycopg.conninfo
import psycopg.sql
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

EMBED_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents(
  id TEXT PRIMARY KEY, name TEXT, kind TEXT, source TEXT, path TEXT,
  created DOUBLE PRECISION, meta JSONB);
CREATE TABLE IF NOT EXISTS chunks(
  id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  idx INTEGER, text TEXT, modality TEXT, loc TEXT, emb vector({EMBED_DIM}));
CREATE TABLE IF NOT EXISTS entities(
  id SERIAL PRIMARY KEY, key TEXT UNIQUE, name TEXT, type TEXT);
CREATE TABLE IF NOT EXISTS mentions(
  entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS relations(
  src INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  dst INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  relation TEXT, chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_mentions_e ON mentions(entity_id);
CREATE INDEX IF NOT EXISTS ix_mentions_c ON mentions(chunk_id);
CREATE INDEX IF NOT EXISTS ix_chunks_emb ON chunks USING hnsw (emb vector_cosine_ops);
"""


def _connect(url):
    """Connect, creating the target database first if it does not exist yet."""
    try:
        return psycopg.connect(url, autocommit=True)
    except psycopg.errors.InvalidCatalogName:
        info = psycopg.conninfo.conninfo_to_dict(url)
        dbname = info.pop("dbname")
        admin = psycopg.connect(**info, dbname="postgres", autocommit=True)
        admin.execute(psycopg.sql.SQL("CREATE DATABASE {}").format(psycopg.sql.Identifier(dbname)))
        admin.close()
        return psycopg.connect(url, autocommit=True)


class Store:
    def __init__(self, url, schema=None):
        if not url:
            raise RuntimeError(
                "MINDMESH_DATABASE_URL is not set. Add e.g. "
                "MINDMESH_DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/mindmesh to .env")
        self.lock = threading.RLock()
        self.version = 0  # bumped on every write; indexes rebuild when it changes
        self.schema = schema
        self.db = _connect(url)
        self.db.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if schema:  # isolated schema (used by tests); `public` stays on the path for the vector type
            self.db.execute(psycopg.sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(psycopg.sql.Identifier(schema)))
            self.db.execute(psycopg.sql.SQL("SET search_path TO {}, public").format(psycopg.sql.Identifier(schema)))
        register_vector(self.db)
        self.db.row_factory = dict_row
        self.db.execute(SCHEMA)

    def drop_schema(self):
        if self.schema:
            self.db.execute(psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(self.schema)))

    def close(self):
        self.db.close()

    # -- documents -------------------------------------------------------
    def add_document(self, name, kind, source="", path="", meta=None):
        doc_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.db.execute("INSERT INTO documents VALUES(%s,%s,%s,%s,%s,%s,%s)",
                            (doc_id, name, kind, source, path, time.time(), json.dumps(meta or {})))
            self.version += 1
        return doc_id

    def set_meta(self, doc_id, meta):
        with self.lock:
            self.db.execute("UPDATE documents SET meta=%s WHERE id=%s", (json.dumps(meta), doc_id))

    def set_path(self, doc_id, path):
        with self.lock:
            self.db.execute("UPDATE documents SET path=%s WHERE id=%s", (path, doc_id))

    def add_chunks(self, doc_id, chunks, embeddings):
        """chunks: list of dicts(text, modality, loc). Returns chunk ids."""
        ids = []
        with self.lock, self.db.transaction():
            for i, (c, e) in enumerate(zip(chunks, embeddings)):
                cid = f"{doc_id}-{i}"
                ids.append(cid)
                self.db.execute("INSERT INTO chunks VALUES(%s,%s,%s,%s,%s,%s,%s)",
                                (cid, doc_id, i, c["text"], c.get("modality", "text"), c.get("loc", ""),
                                 np.asarray(e, dtype=np.float32)))
            self.version += 1
        return ids

    def list_documents(self):
        with self.lock:
            return self.db.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.id)::int AS n_chunks "
                "FROM documents d ORDER BY created DESC").fetchall()

    def get_document(self, doc_id):
        with self.lock:
            return self.db.execute("SELECT * FROM documents WHERE id=%s", (doc_id,)).fetchone()

    def delete_document(self, doc_id):
        with self.lock, self.db.transaction():
            self.db.execute("DELETE FROM documents WHERE id=%s", (doc_id,))  # cascades to chunks/mentions/relations
            self.db.execute("DELETE FROM entities WHERE id NOT IN (SELECT entity_id FROM mentions)")
            self.version += 1

    def all_chunks(self):
        """Chunk rows (without embeddings) for the in-memory BM25 / graph index."""
        with self.lock:
            return self.db.execute(
                "SELECT c.id, c.doc_id, c.idx, c.text, c.modality, c.loc, d.name AS doc_name, d.kind "
                "FROM chunks c JOIN documents d ON d.id=c.doc_id ORDER BY d.created, c.idx").fetchall()

    def vector_search(self, qvec, k=20):
        """Nearest chunks by cosine similarity, computed by pgvector. Returns [(chunk_id, score)]."""
        q = np.asarray(qvec, dtype=np.float32)
        if q.shape[0] != EMBED_DIM:
            raise ValueError(f"Query embedding has {q.shape[0]} dims; the database expects {EMBED_DIM}.")
        with self.lock:
            rows = self.db.execute("SELECT id, 1 - (emb <=> %s) AS score FROM chunks ORDER BY emb <=> %s LIMIT %s",
                                   (q, q, k)).fetchall()
        return [(r["id"], float(r["score"])) for r in rows]

    # -- entities --------------------------------------------------------
    def add_entities(self, chunk_id, entities, relations=()):
        """entities: [(name, type)], relations: [(src_name, relation, dst_name)]."""
        ids = {}
        with self.lock, self.db.transaction():
            for name, etype in entities:
                key = name.lower().strip()
                eid = self.db.execute(
                    "INSERT INTO entities(key,name,type) VALUES(%s,%s,%s) "
                    "ON CONFLICT (key) DO UPDATE SET key=EXCLUDED.key RETURNING id", (key, name, etype)).fetchone()["id"]
                ids[key] = eid
                self.db.execute("INSERT INTO mentions VALUES(%s,%s)", (eid, chunk_id))
            for s, rel, d in relations:
                sk, dk = s.lower().strip(), d.lower().strip()
                if sk in ids and dk in ids:
                    self.db.execute("INSERT INTO relations VALUES(%s,%s,%s,%s)", (ids[sk], ids[dk], rel, chunk_id))
            self.version += 1

    def graph_rows(self):
        with self.lock:
            ents = self.db.execute("SELECT id,name,type FROM entities").fetchall()
            ments = [(r["entity_id"], r["chunk_id"]) for r in self.db.execute("SELECT entity_id, chunk_id FROM mentions")]
            rels = self.db.execute("SELECT src,dst,relation,chunk_id FROM relations").fetchall()
        return ents, ments, rels
