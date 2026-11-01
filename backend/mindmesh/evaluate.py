"""Evaluation of retrieval quality, answer accuracy and citation reliability.

Dataset (eval/dataset.json): list of
  {"question", "gold_docs": [doc names that must be retrieved],
   "answer_terms": [facts that must appear in the answer], "answerable": bool}
Unanswerable questions measure abstention (the system must not fabricate).
"""
import json

from . import config, reasoning

DATASET = config.ROOT / "eval" / "dataset.json"
MODES = ["vector", "bm25", "hybrid", "hybrid_graph"]


def run_eval(index, modes=None, k=6):
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    names = {m["doc_name"] for m in index.metas}
    out = {}
    for mode in modes or MODES:
        rows = []
        for it in items:
            res = reasoning.ask(index, it["question"], mode=mode, k=k)
            got = [s["doc"] for s in res["sources"]]
            gold = it.get("gold_docs", [])
            gold = [g for g in gold if g in names]
            recall = (sum(g in got for g in gold) / len(gold)) if gold else None
            rr = 0.0
            for rank, d in enumerate(got, 1):
                if d in gold:
                    rr = 1 / rank
                    break
            ans = res["answer"].lower()
            abstained = "couldn't find evidence" in ans or "knowledge base is empty" in ans
            if it.get("answerable", True):
                terms = it.get("answer_terms", [])
                acc = sum(t.lower() in ans for t in terms) / len(terms) if terms else None
            else:
                acc = 1.0 if abstained else 0.0
            v = res["verification"]
            rows.append({"question": it["question"], "recall": recall, "rr": rr if gold else None, "accuracy": acc,
                         "citations_valid": not v["invalid_citations"], "support_rate": v["support_rate"],
                         "answerable": it.get("answerable", True), "abstained": abstained})
        out[mode] = {"summary": _summ(rows), "rows": rows}
    return {"k": k, "n": len(items), "results": out}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _summ(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {"recall@k": _mean(r["recall"] for r in rows), "mrr": _mean(r["rr"] for r in rows),
            "answer_accuracy": _mean(r["accuracy"] for r in ans),
            "citation_validity": _mean(float(r["citations_valid"]) for r in ans if not r["abstained"]),
            "citation_support": _mean(r["support_rate"] for r in ans),
            "abstention_correct": _mean(r["accuracy"] for r in unans)}
