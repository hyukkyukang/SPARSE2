"""Full-corpus confirmation (§0.2): re-run the selected configuration and the
references on the whole 8.84M collection, so no headline number rests on C1."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

os.environ.setdefault("JAVA_HOME", "/workspace/SPARSE/dvlsr/tools/jdk-21.0.5+11")
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ.setdefault("PYSERINI_CACHE", "/workspace/SPARSE/dvlsr/pyserini_cache")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.data import Collection, load_queries, qrels_dict
from dvlsr.encoders import Encoder
from dvlsr.metrics import evaluate, bootstrap_ci
from dvlsr.model import Head, segment_max
from dvlsr.retrieval import InvertedIndex
from dvlsr.util import get_logger, gpu_map, save_json, Timer

lg = get_logger("full", "60_full_corpus.log")
CAP = 512


def paths_for(name, kind):
    b = paths.EMB / f"FULL_{name}_{kind}"
    return b.with_name(b.name + "_idx.npy"), b.with_name(b.name + "_val.npy")


def worker(shard, n_shards, name, bs):
    sys.path.insert(0, str(paths.REPO / "scripts"))
    import importlib.util as iu
    spec = iu.spec_from_file_location("e43", paths.REPO / "scripts" / "43_encode_eval.py")
    e43 = iu.module_from_spec(spec); spec.loader.exec_module(e43)
    cfg, head, enc, E = e43.load_ckpt(name)
    layer = cfg["layer"]
    mu, W = whitening.load(paths.ART / "whiten" / f"{cfg['encoder']}_L{layer}_H.npz")
    tf = whitening.Transform("whitened", mu, W, "cuda")
    col = Collection()
    n = len(col)
    b = np.linspace(0, n, n_shards + 1).astype(np.int64)
    lo, hi = int(b[shard]), int(b[shard + 1])
    AI = np.lib.format.open_memmap(paths_for(name, "d")[0], mode="r+")
    AV = np.lib.format.open_memmap(paths_for(name, "d")[1], mode="r+")
    idx = np.arange(lo, hi)
    order = np.argsort([col.off[i + 1] - col.off[i] for i in idx])
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), bs):
            sel = idx[order[s:s + bs]]
            wu = enc.encode_word_units(col.texts(sel), [layer])
            if len(wu.word) == 0:
                continue
            h = tf(wu.states[:, 0].cuda(), normalize=False)
            row = torch.as_tensor(wu.row.astype(np.int64), device="cuda")
            z = head.logits(h.to(E.dtype), E)
            sv = torch.log1p(torch.relu(segment_max(z, row, len(sel))))
            v, i = torch.topk(sv.float(), CAP, dim=1)
            AI[sel] = i.cpu().numpy().astype(np.int32)
            AV[sel] = v.cpu().numpy().astype(np.float16)
            del z, sv
            if s % (bs * 300) == 0:
                print(f"shard {shard}: {s}/{len(order)} "
                      f"{(s+bs)/max(time.time()-t0,1e-9):.0f}/s", flush=True)
    AI.flush(); AV.flush()
    print(f"shard {shard} done {time.time()-t0:.0f}s", flush=True)


def search_full(name, k=1000):
    qi = np.load(paths.EMB / f"D_{name}_q_idx.npy")
    qv = np.load(paths.EMB / f"D_{name}_q_val.npy")
    AI = np.load(paths_for(name, "d")[0], mmap_mode="r")
    AV = np.load(paths_for(name, "d")[1], mmap_mode="r")
    with Timer("index full collection", lg):
        ix = InvertedIndex(np.asarray(AI), np.asarray(AV), 30000, min_val=0.0)
    with Timer("search full collection", lg):
        docs, _ = ix.search(np.asarray(qi), np.asarray(qv, np.float32), 0.0, 0.0,
                            qi.shape[1], CAP, "none", k=k, qchunk=64)
    qids, _ = load_queries(paths.QUERIES_DEV_SMALL)
    np.savez(paths.RUNS / f"full_{name}.npz", qids=np.asarray(qids),
             docs=docs.astype(np.int32))
    qr = {int(a): set(int(x) for x in b)
          for a, b in qrels_dict(paths.QRELS_DEV_SMALL).items()}
    run = {int(q): docs[i] for i, q in enumerate(qids)}
    m, per = evaluate(run, qr)
    m["mrr@10_ci"] = bootstrap_ci(per["mrr@10"])
    m["nnz_d"] = float((np.asarray(AV[::97], np.float32) > 0).sum(1).mean())
    m["nnz_q"] = float((np.asarray(qv, np.float32) > 0).sum(1).mean())
    lg.info(f"FULL {name}: {m}")
    return m


def references_full(k=1000):
    """Re-run BM25 and SPLADE++ at depth k on the full collection for R@1000."""
    sys.path.insert(0, str(paths.REPO / "scripts"))
    import importlib.util as iu
    spec = iu.spec_from_file_location("r3", paths.REPO / "scripts" / "03_reference_runs.py")
    r3 = iu.module_from_spec(spec); spec.loader.exec_module(r3)
    out = {}
    for what, fn in (("bm25", r3.bm25), ("spladepp", r3.splade)):
        p = paths.RUNS / f"{what}_full_top{k}.npz"
        if not p.exists():
            with Timer(f"{what} full @{k}", lg):
                fn(k)
        out[what] = r3.score_run(f"{what}_full_top{k}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=32)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--stage", default="all", choices=["all", "encode", "search", "refs"])
    a = ap.parse_args()
    if a.shard >= 0:
        worker(a.shard, a.n_shards, a.name, a.bs)
        sys.exit(0)
    res = {}
    if a.stage in ("all", "encode"):
        pi, pv = paths_for(a.name, "d")
        if not pi.exists():
            np.lib.format.open_memmap(pi, mode="w+", dtype=np.int32,
                                      shape=(paths.N_PASSAGES, CAP))
            np.lib.format.open_memmap(pv, mode="w+", dtype=np.float16,
                                      shape=(paths.N_PASSAGES, CAP))
        with Timer("encode full collection", lg):
            gpu_map(os.path.abspath(__file__), a.n_shards,
                    ["--name", a.name, "--bs", str(a.bs)], logger=lg, per_gpu=a.per_gpu)
    if a.stage in ("all", "search"):
        res["ours"] = search_full(a.name)
    if a.stage in ("all", "refs"):
        res.update(references_full())
    if res:
        p = paths.RESULTS / "60_full_corpus.json"
        old = json.load(open(p)) if p.exists() else {}
        old.update(res)
        save_json(old, p)
