"""Reference systems on the full collection: BM25 and SPLADE++ (CoCondenser-EnsembleDistil).

Full-collection top-100 per dev query feeds C1 (§0.2); the same machinery is
reused later to evaluate the references on C1 itself.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

os.environ.setdefault("JAVA_HOME", "/workspace/SPARSE/dvlsr/tools/jdk-21.0.5+11")
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ.setdefault("PYSERINI_CACHE", "/workspace/SPARSE/dvlsr/pyserini_cache")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import load_queries, qrels_dict
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("refruns", "03_reference_runs.log")


def save_run(name: str, qids, ranked: list[np.ndarray], scores: list[np.ndarray]):
    n = max(len(r) for r in ranked)
    D = np.full((len(qids), n), -1, dtype=np.int32)
    S = np.zeros((len(qids), n), dtype=np.float32)
    for i, (r, s) in enumerate(zip(ranked, scores)):
        D[i, : len(r)] = r
        S[i, : len(s)] = s
    np.savez(paths.RUNS / f"{name}.npz", qids=np.asarray(qids, np.int64), docs=D, scores=S)
    lg.info(f"saved run {name}: {D.shape}")


def bm25(k: int, k1=0.82, b=0.68, threads=48):
    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
    s.set_bm25(k1, b)
    qids, texts = load_queries(paths.QUERIES_DEV_SMALL)
    ranked, scores = [], []
    t0 = time.time()
    B = 500
    for i in range(0, len(qids), B):
        qs = [str(q) for q in qids[i:i + B]]
        hits = s.batch_search(texts[i:i + B], qs, k=k, threads=threads)
        for q in qs:
            h = hits[q]
            ranked.append(np.asarray([int(x.docid) for x in h], np.int32))
            scores.append(np.asarray([x.score for x in h], np.float32))
        lg.info(f"bm25 {i+B}/{len(qids)} ({time.time()-t0:.0f}s)")
    save_run(f"bm25_full_top{k}", qids, ranked, scores)


def splade(k: int, threads=48):
    from pyserini.search.lucene import LuceneImpactSearcher
    s = LuceneImpactSearcher.from_prebuilt_index(
        "msmarco-v1-passage.splade-pp-ed", query_encoder="SpladePlusPlusEnsembleDistil",
        encoder_type="onnx")
    qids, texts = load_queries(paths.QUERIES_DEV_SMALL)
    ranked, scores = [], []
    t0 = time.time()
    B = 200
    for i in range(0, len(qids), B):
        qs = [str(q) for q in qids[i:i + B]]
        hits = s.batch_search(texts[i:i + B], qs, k=k, threads=threads)
        for q in qs:
            h = hits[q]
            ranked.append(np.asarray([int(x.docid) for x in h], np.int32))
            scores.append(np.asarray([x.score for x in h], np.float32))
        if i % 1000 == 0:
            lg.info(f"splade {i+B}/{len(qids)} ({time.time()-t0:.0f}s)")
    save_run(f"spladepp_full_top{k}", qids, ranked, scores)


def score_run(name):
    from dvlsr.metrics import evaluate, bootstrap_ci
    z = np.load(paths.RUNS / f"{name}.npz")
    docs, qq = z["docs"], z["qids"]
    run = {int(q): docs[i] for i, q in enumerate(qq)}
    qr = {int(k): set(int(x) for x in v) for k, v in qrels_dict(paths.QRELS_DEV_SMALL).items()}
    m, per = evaluate(run, qr)
    m["mrr@10_ci"] = bootstrap_ci(per["mrr@10"])
    lg.info(f"{name}: {m}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all")
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    res = {}
    if a.what in ("all", "bm25"):
        with Timer("BM25 full-collection", lg):
            bm25(a.k)
        res["bm25"] = score_run(f"bm25_full_top{a.k}")
    if a.what in ("all", "splade"):
        with Timer("SPLADE++ full-collection", lg):
            splade(a.k)
        res["spladepp"] = score_run(f"spladepp_full_top{a.k}")
    if res:
        p = paths.RESULTS / "03_reference_full.json"
        old = {}
        if p.exists():
            import json; old = json.load(open(p))
        old.update(res)
        save_json(old, p)
