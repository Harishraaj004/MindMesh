"""Hybrid retrieval: vector + BM25 -> reciprocal-rank fusion -> graph expansion -> diversity rerank."""
import math
from collections import Counter, defaultdict

import numpy as np

from .graph import KnowledgeGraph
from .models import embedder
from .textutil import tokens

RRF_K = 60


class Index:
    def __init__(self, store):
        self.store = store
        self.version = -1
        self.refresh()

    def refresh(self):
        if self.version == self.store.version:
            return
        self.metas = self.store.all_chunks()
        self.by_id = {m["id"]: i for i, m in enumerate(self.metas)}
        # BM25 statistics
        self.tf, self.df, self.len = [], Counter(), []
        for m in self.metas:
            c = Counter(tokens(m["text"]))
            self.tf.append(c)
            self.len.append(sum(c.values()))
            self.df.update(c.keys())
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1
        self.graph = KnowledgeGraph(self.store, {m["id"]: m for m in self.metas})
        self.version = self.store.version

    # -- individual retrievers ------------------------------------------
    def idf(self, term):
        n = len(self.metas)
        return math.log(1 + (n - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))

    def bm25(self, query, k=20, k1=1.5, b=0.75):
        q = tokens(query)
        scores = np.zeros(len(self.metas))
        for t in set(q):
            idf = self.idf(t)
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.len[i] / self.avg))
        return self._top(scores, k)

    def vector(self, query, k=20):
        """Cosine nearest neighbours, computed inside Postgres by pgvector."""
        if not self.metas:
            return []
        rows = self.store.vector_search(embedder.embed([query])[0], k)
        return [(self.by_id[cid], sc) for cid, sc in rows if cid in self.by_id and sc > 0]

    @staticmethod
    def _top(scores, k):
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    # -- fused search -----------------------------------------------------
    def search(self, query, k=8, mode="hybrid_graph", per_doc=3):
        """mode: vector | bm25 | hybrid | hybrid_graph. Returns (hits, trace)."""
        self.refresh()
        trace = {"query": query, "mode": mode, "graph_entities": [], "graph_paths": []}
        if not self.metas:
            return [], trace
        lists = {}
        if mode in ("vector", "hybrid", "hybrid_graph"):
            lists["vector"] = self.vector(query)
        if mode in ("bm25", "hybrid", "hybrid_graph"):
            lists["bm25"] = self.bm25(query)

        fused = defaultdict(float)
        why = defaultdict(set)
        for name, lst in lists.items():
            for rank, (i, _) in enumerate(lst):
                fused[i] += 1.0 / (RRF_K + rank + 1)
                why[i].add(name)

        if mode == "hybrid_graph":
            g = self.graph
            seeds = g.match_query(query)
            if not seeds:  # fall back to entities in the top lexical/vector hits
                top = sorted(fused, key=fused.get, reverse=True)[:3]
                counts = Counter(e for i in top for e in g.c2e.get(self.metas[i]["id"], ()))
                seeds = [e for e, _ in counts.most_common(4)]
            reach = g.expand(seeds, hops=1)
            trace["graph_entities"] = [g.ent[e]["name"] for e in reach]
            for e, (d, via) in reach.items():
                if via is not None:
                    trace["graph_paths"].append(
                        {"from": g.ent[via]["name"], "to": g.ent[e]["name"],
                         "relation": ", ".join(sorted(g.edges[via][e]["relations"])[:1]) or "co-occurs"})
            # Seeds (entities the question is about) count fully; their neighbours only as context.
            gscores = g.chunks_for({e: (1.0 if d == 0 else 0.25) for e, (d, _) in reach.items()})
            ranked = sorted(gscores, key=lambda c: (gscores[c], fused.get(self.by_id[c], 0)), reverse=True)[:20]
            for rank, cid in enumerate(ranked):
                i = self.by_id[cid]
                fused[i] += 1.0 / (RRF_K + rank + 1)  # graph is a full retrieval signal, like vector and BM25
                why[i].add("graph")

        # Diversity rerank: cap chunks per document so one source can't crowd out the others.
        per_doc = max(per_doc, round(k * 0.6))  # a long single source may legitimately dominate
        hits, used = [], Counter()
        for i in sorted(fused, key=fused.get, reverse=True):
            m = self.metas[i]
            if used[m["doc_id"]] >= per_doc:
                continue
            used[m["doc_id"]] += 1
            hits.append({**m, "score": round(fused[i], 5), "via": sorted(why[i])})
            if len(hits) >= k:
                break
        return hits, trace
