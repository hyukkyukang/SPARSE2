"""BM25 hard negatives for Pilot D's training queries (§D.2).

200k MS MARCO train queries with a qrels positive; BM25 top-100 over the full
collection, positives removed. Seven are sampled per query per epoch.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

os.environ.setdefault("JAVA_HOME", "/workspace/SPARSE/dvlsr/tools/jdk-21.0.5+11")
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ.setdefault("PYSERINI_CACHE", "/workspace/SPARSE/dvlsr/pyserini_cache")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import load_queries, load_qrels
from dvlsr.util import get_logger, rng, save_json, Timer

lg = get_logger("negs", "40_mine_negatives.log")
TOPK = 100


def main(n_queries, threads):
    out = paths.PREP / "train_negatives.npz"
    if out.exists():
        lg.info("exists"); return
    qids, qtexts = load_queries(paths.QUERIES_TRAIN)
    qpos = {}
    a, b = load_qrels(paths.QRELS_TRAIN)
    for q, d in zip(a, b):
        qpos.setdefault(int(q), []).append(int(d))
    have = np.array([i for i, q in enumerate(qids) if int(q) in qpos])
    g = rng("train-queries")
    sel = np.sort(g.choice(have, size=min(n_queries, len(have)), replace=False))
    lg.info(f"{len(have)} train queries with qrels; using {len(sel)}")

    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
    s.set_bm25(0.82, 0.68)
    negs = np.full((len(sel), TOPK), -1, np.int32)
    t0 = time.time()
    B = 2000
    for i in range(0, len(sel), B):
        idx = sel[i:i + B]
        qs = [str(int(qids[j])) for j in idx]
        hits = s.batch_search([qtexts[j] for j in idx], qs, k=TOPK, threads=threads)
        for r, j in enumerate(idx):
            pos = set(qpos[int(qids[j])])
            d = [int(x.docid) for x in hits[qs[r]] if int(x.docid) not in pos]
            negs[i + r, : len(d)] = d[:TOPK]
        if i % 20000 == 0:
            el = time.time() - t0
            lg.info(f"{i+B}/{len(sel)} ({el:.0f}s, eta {el/(i+B)*(len(sel)-i-B):.0f}s)")
    pos_arr = np.array([qpos[int(qids[j])][0] for j in sel], np.int32)
    np.savez(out, qidx=sel.astype(np.int32), qids=qids[sel].astype(np.int64),
             pos=pos_arr, negs=negs)
    save_json(dict(n_queries=int(len(sel)),
                   mean_negs=float((negs >= 0).sum(1).mean())),
              paths.RESULTS / "40_negatives.json")
    lg.info(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=paths.N_TRAIN_QUERIES)
    ap.add_argument("--threads", type=int, default=96)
    with Timer("mine BM25 negatives", lg):
        a = ap.parse_args(); main(a.n, a.threads)
