"""Encode C1 (and the dev queries) with the training-free rule, once (§C.2.1).

We store the top-256 entries per passage by the *pre-threshold* max-pooled
similarity p_j, so every (tau, k) cell of the §C.1 grid is a re-truncation of one
stored set of profiles rather than a re-encoding.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.data import Collection, load_queries
from dvlsr.encoders import Encoder
from dvlsr.sparse import profile_maxpool
from dvlsr.util import get_logger, gpu_map, Timer

lg = get_logger("encC1", "30_encode_c1.log")
TOP_D, TOP_Q = 256, 128


def paths_for(enc, layer, rep, kind):
    b = paths.EMB / f"c1_{enc}_L{layer}_{rep}"
    return b.with_name(b.name + f"_{kind}_idx.npy"), b.with_name(b.name + f"_{kind}_val.npy")


def entry_matrix(enc, layer, rep, r1_prefix="doc"):
    WD = paths.ART / "whiten"
    if rep == "R2":
        E = np.load(paths.ART / f"proto_{enc}.npz")["protoA"][:, paths.LAYERS.index(layer)]
        wp = WD / f"{enc}_L{layer}_V_R2.npz"
    else:
        E = np.load(paths.ART / f"r1_{enc}.npz")[r1_prefix]
        wp = WD / f"{enc}_R1{r1_prefix}_V.npz"
    mu, W = whitening.load(wp)
    return whitening.Transform("whitened", mu, W, "cuda")(
        torch.as_tensor(np.asarray(E, np.float32), device="cuda")).half()


def token_tf(enc, layer, query_side=False):
    WD = paths.ART / "whiten"
    tag = "Q" if query_side else "H"
    mu, W = whitening.load(WD / f"{enc}_L{layer}_{tag}.npz")
    return whitening.Transform("whitened", mu, W, "cuda")


def worker(shard, n_shards, enc_name, layer, rep, bs, r1_prefix, qside):
    E = entry_matrix(enc_name, layer, rep, r1_prefix)
    li = paths.LAYERS.index(layer)
    enc = Encoder(enc_name)
    tf = token_tf(enc_name, layer, qside)
    if qside:
        qids, texts = load_queries(paths.QUERIES_DEV_SMALL)
        ids = np.arange(len(texts))
        kind, top = "q", TOP_Q
        getter = lambda sel: [texts[i] for i in sel]
    else:
        pids = np.load(paths.PREP / "c1_pids.npy")
        col = Collection()
        ids = np.arange(len(pids))
        kind, top = "d", TOP_D
        getter = lambda sel: col.texts(pids[sel])
    b = np.linspace(0, len(ids), n_shards + 1).astype(np.int64)
    lo, hi = int(b[shard]), int(b[shard + 1])
    pi, pv = paths_for(enc_name, layer, rep, kind)
    AI = np.lib.format.open_memmap(pi, mode="r+")
    AV = np.lib.format.open_memmap(pv, mode="r+")
    t0 = time.time()
    for s in range(lo, hi, bs):
        sel = np.arange(s, min(s + bs, hi))
        wu = enc.encode_word_units(getter(sel), [layer], is_query=qside,
                                   maxlen=paths.MAXLEN_QRY if qside else paths.MAXLEN_DOC)
        if len(wu.word) == 0:
            continue
        H = tf(wu.states[:, 0].cuda()).half()
        row = torch.as_tensor(wu.row.astype(np.int64), device="cuda")
        P = profile_maxpool(H, row, len(sel), E)
        v, i = torch.topk(P, top, dim=1)
        AI[sel] = i.cpu().numpy().astype(np.int32)
        AV[sel] = v.cpu().numpy().astype(np.float16)
        del P, H
        if (s - lo) % (bs * 100) == 0:
            print(f"shard {shard}: {s-lo}/{hi-lo} "
                  f"{(s-lo+bs)/max(time.time()-t0,1e-9):.0f}/s", flush=True)
    AI.flush(); AV.flush()
    print(f"shard {shard} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=32)
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--rep", default="R2")
    ap.add_argument("--r1-prefix", default="doc")
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--queries", action="store_true")
    a = ap.parse_args()
    if a.shard >= 0:
        worker(a.shard, a.n_shards, a.encoder, a.layer, a.rep, a.bs, a.r1_prefix, a.queries)
    else:
        for qside, kind, top in ((False, "d", TOP_D), (True, "q", TOP_Q)):
            n = (len(np.load(paths.PREP / "c1_pids.npy")) if not qside
                 else len(load_queries(paths.QUERIES_DEV_SMALL)[0]))
            pi, pv = paths_for(a.encoder, a.layer, a.rep, kind)
            if not pi.exists():
                np.lib.format.open_memmap(pi, mode="w+", dtype=np.int32, shape=(n, top))
                np.lib.format.open_memmap(pv, mode="w+", dtype=np.float16, shape=(n, top))
            extra = ["--encoder", a.encoder, "--layer", str(a.layer), "--rep", a.rep,
                     "--r1-prefix", a.r1_prefix, "--bs", str(a.bs)]
            if qside:
                extra.append("--queries")
            ns = a.n_shards if not qside else 8
            with Timer(f"encode C1 {kind} ({n})", lg):
                gpu_map(os.path.abspath(__file__), ns, extra, logger=lg,
                        per_gpu=a.per_gpu)
