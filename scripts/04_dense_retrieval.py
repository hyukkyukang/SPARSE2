"""Exact dense retrieval over the full collection (sanity check + C1 candidates).

The whole fp16 corpus matrix (8.84M x 768 = 13.6 GB) fits on one A100, so this is
exact brute-force search, not an approximate index.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import load_queries, qrels_dict
from dvlsr.encoders import Encoder
from dvlsr.metrics import evaluate, bootstrap_ci
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("dense", "04_dense_retrieval.log")


def search(enc_name, k=1000, qchunk=256, subset=None, queries=None, tag="full"):
    C = np.load(paths.EMB / f"corpus_{enc_name}.npy", mmap_mode="r")
    if queries is None:
        qids, qtexts = load_queries(paths.QUERIES_DEV_SMALL)
    else:
        qids, qtexts = queries
    enc = Encoder(enc_name)
    Qv = torch.from_numpy(enc.encode_pooled(qtexts, is_query=True)).cuda()
    del enc
    torch.cuda.empty_cache()
    with Timer(f"load corpus {enc_name} to gpu", lg):
        if subset is None:
            D = torch.from_numpy(np.ascontiguousarray(C)).cuda()
            ids = None
        else:
            D = torch.from_numpy(np.ascontiguousarray(C[subset])).cuda()
            ids = torch.from_numpy(np.asarray(subset, np.int64)).cuda()
    docs = np.zeros((len(qids), k), np.int32)
    scr = np.zeros((len(qids), k), np.float32)
    with Timer(f"search {enc_name} {tag} ({D.shape[0]} docs)", lg):
        for s in range(0, len(qids), qchunk):
            q = Qv[s:s + qchunk]
            sc = (q @ D.T).float()
            v, i = torch.topk(sc, min(k, D.shape[0]), dim=1)
            if ids is not None:
                i = ids[i]
            docs[s:s + len(q), : v.shape[1]] = i.cpu().numpy()
            scr[s:s + len(q), : v.shape[1]] = v.cpu().numpy()
    del D
    torch.cuda.empty_cache()
    return qids, docs, scr


def main(enc_name, k):
    qids, docs, scr = search(enc_name, k)
    np.savez(paths.RUNS / f"dense_{enc_name}_full_top{k}.npz",
             qids=np.asarray(qids), docs=docs, scores=scr)
    qr = {int(a): set(int(x) for x in b) for a, b in qrels_dict(paths.QRELS_DEV_SMALL).items()}
    run = {int(q): docs[i] for i, q in enumerate(qids)}
    m, per = evaluate(run, qr)
    m["mrr@10_ci"] = bootstrap_ci(per["mrr@10"])
    lg.info(f"dense {enc_name} full collection: {m}")
    p = paths.RESULTS / "04_dense_full.json"
    old = {}
    if p.exists():
        import json; old = json.load(open(p))
    old[enc_name] = m
    save_json(old, p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--k", type=int, default=1000)
    a = ap.parse_args()
    main(a.encoder, a.k)
