"""Entity extraction and the knowledge graph.

Extraction is LLM-driven when a model is configured (typed relations), and a
heuristic extractor otherwise (capitalised phrases, acronyms, code identifiers;
relations are co-occurrence within a chunk).
"""
import re
from collections import defaultdict, deque

from . import config
from .models import llm
from .textutil import STOP

CAP_PHRASE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:[ -][A-Z][a-zA-Z0-9]+){0,3})\b")
ACRONYM = re.compile(r"\b([A-Z]{2,6}[0-9]?)\b")
CODE_ID = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+|[a-z]+(?:_[a-z0-9]+){1,3})\b")
GENERIC = STOP | {"before", "after", "while", "dear", "until", "since", "though", "because", "during", "now", "but", "the", "this", "it", "we", "figure", "table", "note", "notes", "todo", "see", "following",
                  "however", "also", "one", "two", "first", "second", "section", "chapter", "page", "http", "https"}


def _clean(name):
    name = name.strip(" -")
    words = name.split()
    while words and words[0].lower() in GENERIC:
        words = words[1:]
    return " ".join(words)


def heuristic_entities(text, is_code=False):
    found = {}
    for m in CAP_PHRASE.finditer(text):
        n = _clean(m.group(1))
        if len(n) > 2 and n.lower() not in GENERIC:
            found.setdefault(n.lower(), (n, "concept"))
    for m in ACRONYM.finditer(text):
        n = m.group(1)
        if n.lower() not in GENERIC:
            found.setdefault(n.lower(), (n, "acronym"))
    if is_code:
        for m in CODE_ID.finditer(text):
            found.setdefault(m.group(1).lower(), (m.group(1), "code"))
    return list(found.values())[:25]


LLM_SYS = ("Extract entities and relations from the text for a knowledge graph. Respond as JSON: "
           '{"entities":[{"name":str,"type":str}],"relations":[{"src":str,"relation":str,"dst":str}]}. '
           "Use at most 12 entities; names must appear in the text; relations use short verb phrases.")


def extract(text, is_code=False):
    """Returns (entities[(name,type)], relations[(src,rel,dst)])."""
    if llm.available and config.ENTITY_EXTRACTION == "llm":
        try:
            j = llm.json(LLM_SYS, text[:4000])
            ents = [(e["name"], e.get("type", "concept")) for e in j.get("entities", []) if e.get("name")]
            rels = [(r["src"], r["relation"], r["dst"]) for r in j.get("relations", [])
                    if r.get("src") and r.get("dst")]
            if ents:
                return ents, rels
        except Exception:
            pass
    return heuristic_entities(text, is_code), []


class KnowledgeGraph:
    """In-memory view built from the entity tables."""

    def __init__(self, store, chunk_meta):
        ents, mentions, rels = store.graph_rows()
        self.ent = {e["id"]: e for e in ents}
        self.chunk_meta = chunk_meta  # chunk_id -> meta (doc_id, doc_name, ...)
        self.e2c = defaultdict(set)
        self.c2e = defaultdict(set)
        for eid, cid in mentions:
            if cid in chunk_meta:
                self.e2c[eid].add(cid)
                self.c2e[cid].add(eid)
        self.edges = defaultdict(lambda: defaultdict(lambda: {"weight": 0, "relations": set(), "chunks": set()}))
        for cid, eids in self.c2e.items():  # co-occurrence edges
            eids = sorted(eids)
            for i, a in enumerate(eids):
                for b in eids[i + 1:]:
                    for x, y in ((a, b), (b, a)):
                        self.edges[x][y]["weight"] += 1
                        self.edges[x][y]["chunks"].add(cid)
        for r in rels:  # typed relations from the LLM extractor
            for x, y in ((r["src"], r["dst"]), (r["dst"], r["src"])):
                if x in self.ent and y in self.ent:
                    e = self.edges[x][y]
                    e["weight"] += 2
                    e["relations"].add(r["relation"] if x == r["src"] else f"<- {r['relation']}")
                    e["chunks"].add(r["chunk_id"])

    def docs_of(self, eid):
        return {self.chunk_meta[c]["doc_id"] for c in self.e2c[eid]}

    def match_query(self, query):
        q = " " + re.sub(r"[^a-z0-9_ ]", " ", query.lower()) + " "
        hits = []
        for eid, e in self.ent.items():
            n = e["name"].lower()
            if len(n) > 2 and f" {n} " in q and eid in self.e2c:
                hits.append(eid)
        # Alias resolution: a question about "ron" should also reach the fuller entity "Ron Weasley".
        qt = {t for t in q.split() if t not in GENERIC and len(t) > 2}
        alias_tokens = {self.ent[h]["name"].lower() for h in hits if " " not in self.ent[h]["name"]}
        if len(qt) <= 3:
            alias_tokens |= qt
        if alias_tokens:
            alias = sorted(((len(self.e2c[eid]), eid) for eid, e in self.ent.items()
                            if eid in self.e2c and eid not in hits and len(e["name"].split()) > 1
                            and len(self.e2c[eid]) >= 3 and e["name"].split()[0].lower() not in GENERIC
                            and alias_tokens & set(e["name"].lower().split())), reverse=True)
            hits += [eid for _, eid in alias[:4]]
        return hits

    def cross_doc_bridges(self, limit=8):
        """Entities that appear in 2+ documents: the links that connect separate sources."""
        out = []
        for eid in self.e2c:
            docs = self.docs_of(eid)
            if len(docs) > 1:
                out.append({"entity": self.ent[eid]["name"], "type": self.ent[eid]["type"],
                            "docs": sorted(self.chunk_meta[next(c for c in self.e2c[eid]
                                           if self.chunk_meta[c]["doc_id"] == d)]["doc_name"] for d in docs)})
        out.sort(key=lambda x: -len(x["docs"]))
        return out[:limit]

    def expand(self, seeds, hops=1, max_entities=25):
        """BFS over weighted edges. Returns {entity_id: (distance, via_entity_id)}."""
        seen = {s: (0, None) for s in seeds}
        dq = deque((s, 0) for s in seeds)
        while dq:
            cur, d = dq.popleft()
            if d >= hops:
                continue
            nbrs = sorted(self.edges[cur].items(), key=lambda kv: -kv[1]["weight"])[:8]
            for nb, _ in nbrs:
                if nb not in seen and len(seen) < max_entities:
                    seen[nb] = (d + 1, cur)
                    dq.append((nb, d + 1))
        return seen

    def chunks_for(self, weights):
        """weights: {entity_id: weight}. Chunks score by the (rarity-adjusted) weight of the entities they mention."""
        scores = defaultdict(float)
        for eid, w in weights.items():
            for cid in self.e2c[eid]:
                scores[cid] += w / (1 + len(self.e2c[eid]) ** 0.5)  # rarer entities count more
        return scores

    def export(self, max_nodes=70):
        deg = {eid: sum(v["weight"] for v in self.edges[eid].values()) for eid in self.e2c}
        keep = sorted(deg, key=lambda e: (-len(self.docs_of(e)), -deg[e]))[:max_nodes]
        ks = set(keep)
        docs_by_id = {}
        nodes = []
        for eid in keep:
            ds = sorted({self.chunk_meta[c]["doc_name"] for c in self.e2c[eid]})
            nodes.append({"id": eid, "label": self.ent[eid]["name"], "type": self.ent[eid]["type"],
                          "docs": ds, "bridge": len(ds) > 1, "mentions": len(self.e2c[eid])})
        links, seen = [], set()
        for a in keep:  # keep each node's 3 strongest edges so the picture stays readable
            top = sorted(((b, v) for b, v in self.edges[a].items() if b in ks),
                         key=lambda kv: -kv[1]["weight"])[:3]
            for b, v in top:
                if (min(a, b), max(a, b)) not in seen:
                    seen.add((min(a, b), max(a, b)))
                    links.append({"source": a, "target": b, "weight": v["weight"],
                                  "relation": ", ".join(sorted(v["relations"])[:2])})
        return {"nodes": nodes, "links": links}
