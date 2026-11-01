"""Multi-step reasoning: plan -> retrieve -> build context -> generate -> verify citations."""
import math
import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .models import llm
from .retrieval import Index
from .textutil import sentences, tokens

CITE_RE = re.compile(r"\[S(\d+)\]")
VISUAL_RE = re.compile(r"\b(diagram|visuali[sz]e|draw|illustrate|sketch|show me)\b", re.I)


DEFINE_RE = re.compile(r"^\s*(?:who|what)\s+(?:is|was|are|were)\s+(?:an?\s+|the\s+)?(.+?)\s*\??\s*$", re.I)


def plan(question):
    """Split a question into search queries. Definitional questions ("who is X") are expanded so that
    retrieval finds passages that *describe* X (identity, relationships, role), not just any mention of X."""
    m = DEFINE_RE.match(question)
    if llm.available:
        try:
            hint = (" This is a 'who/what is X' question: cover X's identity and full name, X's relationships "
                    "and family, and X's role, traits or description." if m else "")
            j = llm.json('Break the question into 1-3 standalone search queries that together cover it. '
                         'Respond as JSON {"queries":[str]}. Use 1 query if it is simple.' + hint, question)
            qs = [q for q in j.get("queries", []) if isinstance(q, str) and q.strip()]
            if qs:  # the user's own wording is always searched too
                return ([question] + [q for q in qs if q.lower() != question.lower()])[:3], "llm"
        except Exception as e:
            how = f"heuristic; LLM planning failed: {type(e).__name__}"
        else:
            how = "heuristic"
    else:
        how = "heuristic"
    if m:
        x = m.group(1)
        return [question, f"{x} full name family relationship friend", f"{x} description role personality"], how
    parts = re.split(r"\s*(?:;|\?\s+|\band also\b|\bas well as\b|\bversus\b|\bvs\.?\b)\s*", question, flags=re.I)
    parts = [p.strip(" ?,.") for p in parts if len(tokens(p)) >= 2]
    qs = [question] + [p for p in parts if p.lower() != question.lower().strip(" ?")]
    return qs[:3], how


def gather(index: Index, subqueries, mode, k):
    traces, seen = [], {}
    for sq in subqueries:
        h, tr = index.search(sq, k=k, mode=mode)
        traces.append(tr)
        for x in h:
            if x["id"] not in seen or x["score"] > seen[x["id"]]["score"]:
                seen[x["id"]] = x
    return sorted(seen.values(), key=lambda x: -x["score"])[:k + 2], traces


def build_context(hits):
    return [{"id": f"S{n}", "chunk_id": h["id"], "doc_id": h["doc_id"], "doc": h["doc_name"], "kind": h["kind"],
             "loc": h["loc"], "text": h["text"], "score": h["score"], "via": h["via"]}
            for n, h in enumerate(hits, 1)]


def bridges_for(index: Index, sources):
    """Entities shared by 2+ retrieved documents: the cross-source links behind an answer."""
    g = index.graph
    per_ent = {}
    for s in sources:
        for e in g.c2e.get(s["chunk_id"], ()):
            per_ent.setdefault(e, {})[s["doc"]] = s["id"]
    out = [{"entity": g.ent[e]["name"], "docs": list(d), "sources": sorted(d.values())}
           for e, d in per_ent.items() if len(d) > 1]
    out.sort(key=lambda b: (-len(b["docs"]), b["entity"]))
    return out[:6]


def generate_llm(question, sources, feedback=None):
    ctx = "\n\n".join(f"[{s['id']}] ({s['doc']}, {s['loc']})\n{s['text']}" for s in sources)
    sys = ("You answer questions using only the numbered sources provided. "
           "Structure: first, ONE or TWO sentences that directly answer the question (for 'who/what is X': what or who "
           "X is, and their role and key relationships). Then 3-6 short bullet points of distinct supporting facts. "
           "Synthesize across passages and across documents; do not narrate scenes one by one and do not paste long quotes. "
           "Cite every sentence with its source tag like [S1] or [S2][S3]; never cite a tag that was not provided. "
           "State only what the sources support; if they do not state something the question needs, say so plainly "
           "instead of guessing. If the sources do not answer the question at all, say that.")
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}"}]
    if feedback:
        msgs.append({"role": "user", "content": feedback})
    return llm.chat(msgs)


def generate_local(index: Index, question, sources, bridges, max_sents=6):
    """Deterministic extractive answerer: picks best-supported sentences and cites them."""
    qt = set(tokens(VISUAL_RE.sub(" ", question))) - {"show", "tell", "explain", "diagram", "much", "many"}
    q_mass = sum(index.idf(t) for t in qt) or 1.0
    cands = []
    for rank, s in enumerate(sources):
        for sent in (re.sub(r"^#+\s*[^.:]*:\s*", "", x) for x in sentences(s["text"])):
            st = set(tokens(sent))
            hit = qt & st
            if hit:
                cands.append((sum(index.idf(t) for t in hit) / (len(st) + 10) ** 0.5 + 0.15 / (rank + 1), s, sent))
    cands.sort(key=lambda c: -c[0])
    covered = set().union(*(qt & set(tokens(c[2])) for c in cands[:3])) if cands else set()
    if not cands or sum(index.idf(t) for t in covered) / q_mass < 0.3:  # top evidence covers too little of the question
        return "I couldn't find evidence for this in your knowledge base. Try rephrasing, or upload material that covers it."
    cands.sort(key=lambda c: -c[0])
    picked, per_doc, seen = [], {}, set()
    for _, s, sent in cands:
        if sent in seen or per_doc.get(s["doc"], 0) >= 3:
            continue
        seen.add(sent)
        per_doc[s["doc"]] = per_doc.get(s["doc"], 0) + 1
        picked.append((s, sent))
        if len(picked) >= max_sents:
            break
    picked.sort(key=lambda p: int(p[0]["id"][1:]))
    lines = ["From your sources:"] + [f"- {sent} [{s['id']}]" for s, sent in picked]
    if bridges:
        lines.append("\nHow your sources connect:")
        for b in bridges[:3]:
            lines.append(f"- \"{b['entity']}\" appears in {' and '.join(b['docs'])} " + "".join(f"[{t}]" for t in b["sources"]))
    return "\n".join(lines)


def verify(answer, sources):
    """Check every cited tag exists and the cited text lexically supports the statement."""
    by_id = {s["id"]: s for s in sources}
    invalid, checked, supported, flagged = set(), 0, 0, []
    # Split into statements, but keep a trailing citation ("... Granger. [S1][S7]") attached to its sentence.
    for line in re.split(r"(?<=[.!?])\s+(?!\[S\d)|\n+", answer):
        tags = CITE_RE.findall(line)
        if not tags:
            continue
        invalid |= {f"S{t}" for t in tags if f"S{t}" not in by_id}
        body = set(tokens(CITE_RE.sub("", line)))
        valid = [by_id[f"S{t}"] for t in tags if f"S{t}" in by_id]
        link = re.match(r'^- "(.+?)" appears in ', line)
        if link:  # connection claim: the entity must literally occur in every cited passage
            checked += 1
            if valid and all(link.group(1).lower() in v["text"].lower() for v in valid):
                supported += 1
            else:
                flagged.append({"statement": CITE_RE.sub("", line).strip(" -"), "support": 0.0})
            continue
        if len(body) < 3:
            continue
        checked += 1
        # A synthesized sentence may combine several cited passages, so measure against all of them together,
        # with a stricter bar when more than one source is cited.
        cited_vocab = set().union(*(set(tokens(v["text"])) for v in valid)) if valid else set()
        best = len(body & cited_vocab) / len(body)
        if best >= (0.6 if len(valid) > 1 else 0.5):
            supported += 1
        else:
            flagged.append({"statement": CITE_RE.sub("", line).strip(" -"), "support": round(best, 2)})
    uncited = [l for l in answer.split("\n") if len(tokens(l)) > 8 and not CITE_RE.search(l)
               and not l.rstrip().endswith(":") and "couldn't find evidence" not in l]
    return {"invalid_citations": sorted(invalid), "statements_checked": checked, "statements_supported": supported,
            "support_rate": round(supported / checked, 2) if checked else None,
            "flagged": flagged, "uncited_lines": len(uncited)}


class State(TypedDict, total=False):
    question: str
    mode: str
    k: int
    subqueries: list
    traces: list
    paths: list
    hits: list
    sources: list
    bridges: list
    answer: str
    verification: dict
    attempts: int
    llm_ok: bool
    feedback: str
    steps: Annotated[list, operator.add]  # each node appends; the UI renders this as the reasoning trace


MAX_ATTEMPTS = 2


def build_graph(index: Index):
    """plan -> retrieve -> context -> generate -> verify -(unsupported + LLM)-> generate (once) -> END"""

    def n_plan(st):
        qs, how = plan(st["question"])
        return {"subqueries": qs, "steps": [{"step": "plan", "data": qs, "detail":
                f"{len(qs)} sub-quer{'y' if len(qs) == 1 else 'ies'} ({how})"}]}

    def n_retrieve(st):
        hits, traces = gather(index, st["subqueries"], st["mode"], st["k"])
        paths = [p for t in traces for p in t["graph_paths"]][:10]
        steps = [{"step": "retrieve", "detail": f"{len(hits)} passages from "
                  f"{len({h['doc_id'] for h in hits})} document(s) [{st['mode']}]"}]
        if st["mode"] == "hybrid_graph":
            steps.append({"step": "graph", "detail": f"{len(paths)} relationship hops explored", "data": paths})
        return {"hits": hits, "traces": traces, "paths": paths, "steps": steps}

    def n_context(st):
        sources = build_context(st["hits"])
        return {"sources": sources, "bridges": bridges_for(index, sources)}

    def n_generate(st):
        sources, steps, llm_ok = st["sources"], [], False
        if not sources:
            answer = "Your knowledge base is empty. Upload some documents first."
        elif llm.available:
            try:
                answer, llm_ok = generate_llm(st["question"], sources, st.get("feedback")), True
            except Exception as e:  # never fail the request because the model call failed
                prev = st.get("answer")  # a failed revision must never replace a real answer
                answer = prev or generate_local(index, st["question"], sources, st["bridges"])
                steps.append({"step": "warn", "detail": f"LLM call failed ({type(e).__name__}: {str(e)[:80]}); "
                                                        + ("kept the previous answer" if prev else "used the local answerer")})
        else:
            answer = generate_local(index, st["question"], sources, st["bridges"])
        attempts = st.get("attempts", 0) + 1
        steps.append({"step": "generate", "detail": llm.name + (f" (revision {attempts - 1})" if attempts > 1 else "")})
        return {"answer": answer, "attempts": attempts, "llm_ok": llm_ok, "steps": steps}

    def n_verify(st):
        v = verify(st["answer"], st["sources"])
        fb = ""
        if v["invalid_citations"] or v["flagged"]:
            bad = "; ".join(f'"{f["statement"][:120]}"' for f in v["flagged"][:4])
            fb = ("Your previous answer failed citation checking. "
                  + (f"Invalid tags: {', '.join(v['invalid_citations'])}. " if v["invalid_citations"] else "")
                  + (f"Statements not supported by the cited source: {bad}. " if bad else "")
                  + "Rewrite it using only facts stated in the sources, with correct tags.")
        return {"verification": v, "feedback": fb, "steps": [{"step": "verify", "detail":
                f"{v['statements_supported']}/{v['statements_checked']} cited statements supported"}]}

    def route(st):
        retry = st.get("llm_ok") and st.get("feedback") and st.get("attempts", 0) < MAX_ATTEMPTS
        return "generate" if retry else END

    g = StateGraph(State)
    for name, fn in (("plan", n_plan), ("retrieve", n_retrieve), ("context", n_context),
                     ("generate", n_generate), ("verify", n_verify)):
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "context")
    g.add_edge("context", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route, {"generate": "generate", END: END})
    return g.compile()


_compiled = {}


def ask(index: Index, question, mode="hybrid_graph", k=8):
    index.refresh()
    graph = _compiled.get(id(index)) or _compiled.setdefault(id(index), build_graph(index))
    st = graph.invoke({"question": question, "mode": mode, "k": k, "steps": []})
    return {"question": question, "answer": st["answer"], "sources": st["sources"],
            "cited": sorted({f"S{t}" for t in CITE_RE.findall(st["answer"])}),
            "bridges": st["bridges"], "graph_paths": st["paths"], "verification": st["verification"],
            "steps": st["steps"], "wants_visual": bool(VISUAL_RE.search(question))}
