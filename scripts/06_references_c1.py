"""Reference systems evaluated on C1: BM25, the dense encoder, and SPLADE++ (§C.1).

BM25 gets its own Lucene index over C1. SPLADE++ is scored exactly (dense-vector
sparse dot product over C1) rather than through a quantised impact index, so the
comparison is not disadvantaged by quantisation; its full-collection Pyserini
number (0.3827) is reported alongside as the calibration point.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np
import torch

os.environ.setdefault("JAVA_HOME", paths_java := "/workspace/SPARSE/dvlsr/tools/jdk-21.0.5+11")
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import Collection, load_queries, qrels_dict
from dvlsr.metrics import evaluate, bootstrap_ci
from dvlsr.util import get_logger, save_json, Timer, gpu_map

lg = get_logger("refC1", "06_references_c1.log")


def c1():
    return np.load(paths.PREP / "c1_pids.npy")


def write_jsonl():
    d = paths.PREP / "c1_jsonl"
    d.mkdir(exist_ok=True)
    f = d / "docs.jsonl"
    if f.exists():
        return d
    col = Collection()
    pids = c1()
    with open(f, "w") as fo:
        for p in pids:
            fo.write(json.dumps({"id": str(int(p)), "contents": col[int(p)]}) + "\n")
    lg.info(f"wrote {f}")
    return d


def bm25_c1(k=1000, k1=0.82, b=0.68, threads=48):
    idx = paths.PREP / "c1_lucene"
    if not idx.exists():
        d = write_jsonl()
        with Timer("index C1 with Lucene", lg):
            subprocess.run([sys.executable, "-m", "pyserini.index.lucene",
                            "--collection", "JsonCollection", "--input", str(d),
                            "--index", str(idx), "--generator", "DefaultLuceneDocumentGenerator",
                            "--threads", str(threads), "--storeRaw"],
                           check=True, capture_output=True)
    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher(str(idx))
    s.set_bm25(k1, b)
    qids, texts = load_queries(paths.QUERIES_DEV_SMALL)
    docs = np.full((len(qids), k), -1, np.int32)
    with Timer("BM25 search on C1", lg):
        for i in range(0, len(qids), 500):
            qs = [str(q) for q in qids[i:i + 500]]
            hits = s.batch_search(texts[i:i + 500], qs, k=k, threads=threads)
            for r, q in enumerate(qs):
                h = hits[q]
                docs[i + r, : len(h)] = [int(x.docid) for x in h]
    np.savez(paths.RUNS / "bm25_c1.npz", qids=np.asarray(qids), docs=docs)
    return "bm25_c1"


def splade_worker(shard, n_shards, bs):
    from dvlsr.encoders import SpladeEncoder
    pids = c1()
    b = np.linspace(0, len(pids), n_shards + 1).astype(np.int64)
    lo, hi = int(b[shard]), int(b[shard + 1])
    col = Collection()
    sp = SpladeEncoder()
    TOP = 256
    AI = np.lib.format.open_memmap(paths.EMB / "c1_splade_idx.npy", mode="r+")
    AV = np.lib.format.open_memmap(paths.EMB / "c1_splade_val.npy", mode="r+")
    t0 = time.time()
    for s in range(lo, hi, bs):
        sel = np.arange(s, min(s + bs, hi))
        reps = sp.encode(col.texts(pids[sel]))
        for k, (ii, vv) in enumerate(reps):
            o = np.argsort(-vv)[:TOP]
            AI[sel[k], : len(o)] = ii[o]
            AV[sel[k], : len(o)] = vv[o]
        if (s - lo) % (bs * 100) == 0:
            print(f"shard {shard}: {s-lo}/{hi-lo} "
                  f"{(s-lo+bs)/max(time.time()-t0,1e-9):.0f}/s", flush=True)
    AI.flush(); AV.flush()
    print(f"shard {shard} done", flush=True)


def splade_c1(k=1000, n_shards=24, per_gpu=3):
    pids = c1()
    pi = paths.EMB / "c1_splade_idx.npy"
    if not pi.exists():
        np.lib.format.open_memmap(pi, mode="w+", dtype=np.int32, shape=(len(pids), 256))
        np.lib.format.open_memmap(paths.EMB / "c1_splade_val.npy", mode="w+",
                                  dtype=np.float16, shape=(len(pids), 256))
        with Timer("encode C1 with SPLADE++", lg):
            gpu_map(os.path.abspath(__file__), n_shards, ["--what", "splade-shard"],
                    logger=lg, per_gpu=per_gpu)
    from dvlsr.encoders import SpladeEncoder
    sp = SpladeEncoder()
    qids, texts = load_queries(paths.QUERIES_DEV_SMALL)
    qreps = sp.encode(texts, maxlen=paths.MAXLEN_QRY)
    del sp
    torch.cuda.empty_cache()
    import scipy.sparse as sps_
    AI = np.load(pi, mmap_mode="r"); AV = np.load(paths.EMB / "c1_splade_val.npy", mmap_mode="r")
    nV = 30522
    docs = _sparse_search(AI, AV, qreps, nV, len(pids), k)
    np.savez(paths.RUNS / "spladepp_c1.npz", qids=np.asarray(qids),
             docs=pids[docs].astype(np.int32))
    return "spladepp_c1"


def _sparse_search(AI, AV, qreps, nV, n_docs, k, qbatch=256, dbatch=400_000):
    """Exact sparse dot product: query CSR x doc CSR^T, batched both ways."""
    import scipy.sparse as sp
    nq = len(qreps)
    best_v = np.full((nq, k), -np.inf, np.float32)
    best_i = np.zeros((nq, k), np.int64)
    rows, cols, vals = [], [], []
    for i, (ii, vv) in enumerate(qreps):
        rows.append(np.full(len(ii), i)); cols.append(ii); vals.append(vv)
    Q = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nq, nV), dtype=np.float32)
    for d0 in range(0, n_docs, dbatch):
        d1 = min(d0 + dbatch, n_docs)
        idx = np.asarray(AI[d0:d1]); val = np.asarray(AV[d0:d1], np.float32)
        m = val > 0
        indptr = np.zeros(d1 - d0 + 1, np.int64); indptr[1:] = np.cumsum(m.sum(1))
        D = sp.csr_matrix((val[m], idx[m], indptr), shape=(d1 - d0, nV))
        S = (Q @ D.T).toarray()
        cat_v = np.concatenate([best_v, S], 1)
        cat_i = np.concatenate([best_i, np.arange(d0, d1)[None, :].repeat(nq, 0)], 1)
        o = np.argpartition(-cat_v, k - 1, axis=1)[:, :k]
        r = np.take_along_axis(cat_v, o, 1)
        oo = np.argsort(-r, axis=1)
        best_v = np.take_along_axis(r, oo, 1)
        best_i = np.take_along_axis(np.take_along_axis(cat_i, o, 1), oo, 1)
        lg.info(f"sparse search {d1}/{n_docs}")
    return best_i




def score(name, tag=None):
    z = np.load(paths.RUNS / f"{name}.npz")
    run = {int(q): z["docs"][i] for i, q in enumerate(z["qids"])}
    qr = {int(a): set(int(x) for x in b) for a, b in qrels_dict(paths.QRELS_DEV_SMALL).items()}
    m, per = evaluate(run, qr)
    m["mrr@10_ci"] = bootstrap_ci(per["mrr@10"])
    np.savez(paths.RUNS / f"{name}_perq.npz", qids=per["qids"],
             **{k2: v for k2, v in per.items() if k2 != "qids"})
    lg.info(f"{tag or name}: {m}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=24)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--encoder", default="e5")
    a = ap.parse_args()
    if a.what == "splade-shard":
        splade_worker(a.shard, a.n_shards, a.bs)
        sys.exit(0)
    res = {}
    p = paths.RESULTS / "06_references_c1.json"
    if p.exists():
        res = json.load(open(p))
    if a.what in ("all", "bm25"):
        res["bm25"] = score(bm25_c1())
    if a.what in ("all", "splade"):
        res["spladepp"] = score(splade_c1())
    if a.what in ("all", "dense"):
        sys.path.insert(0, str(paths.REPO / "scripts"))
        import importlib.util as iu
        spec = iu.spec_from_file_location("d4", paths.REPO / "scripts" / "04_dense_retrieval.py")
        d4 = iu.module_from_spec(spec); spec.loader.exec_module(d4)
        pids = c1()
        qids, docs, scr = d4.search(a.encoder, k=1000, subset=pids, tag="C1")
        np.savez(paths.RUNS / f"dense_{a.encoder}_c1.npz", qids=np.asarray(qids),
                 docs=docs, scores=scr)
        res[f"dense_{a.encoder}"] = score(f"dense_{a.encoder}_c1")
    save_json(res, p)
