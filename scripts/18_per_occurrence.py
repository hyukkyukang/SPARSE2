"""Per-occurrence prototype states for the winning encoder at the top-2 layers.

Pilot A only needs the k=50 means, so it accumulates sums and stores nothing else.
Pilot B needs the k-curve (k = 1..50 over two disjoint occurrence sets) and the
multi-prototype variant, which need the individual occurrence states -- but only
for one encoder and two layers, so the 9.2 GB is affordable where 45 GB was not.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import Collection
from dvlsr.encoders import Encoder
from dvlsr.util import get_logger, gpu_map, Timer

lg = get_logger("perocc", "18_per_occurrence.log")


def store_path(enc, layers):
    return paths.ART / f"occ_{enc}_L{'-'.join(map(str, layers))}.npy"


def pair_index():
    z = np.load(paths.ART / "vocab.npz")
    occ = z["occ_pid"]
    nV, K = occ.shape
    j = np.repeat(np.arange(nV, dtype=np.int64), K)
    slot = np.tile(np.arange(K, dtype=np.int64), nV)
    pid = occ.reshape(-1).astype(np.int64)
    ok = pid >= 0
    key = pid[ok] * nV + j[ok]
    o = np.argsort(key)
    return key[o], slot[ok][o], j[ok][o], nV, K, np.unique(pid[ok])


def worker(shard, n_shards, enc_name, layers, bs):
    key, slot, jj_all, nV, K, upid = pair_index()
    b = np.linspace(0, len(upid), n_shards + 1).astype(np.int64)
    mine = upid[b[shard]:b[shard + 1]]
    z = np.load(paths.ART / "vocab.npz")
    inv = {str(w): i for i, w in enumerate(z["words"])}
    col = Collection()
    enc = Encoder(enc_name)
    arr = np.lib.format.open_memmap(store_path(enc_name, layers), mode="r+")
    key_t = torch.from_numpy(key).cuda()
    slot_t = torch.from_numpy(slot).cuda()
    lens = np.asarray([col.off[i + 1] - col.off[i] for i in mine])
    order = np.argsort(lens)
    t0 = time.time()
    for s in range(0, len(order), bs):
        pids = mine[order[s:s + bs]]
        wu = enc.encode_word_units(col.texts(pids), layers, keep=lambda w: w in inv)
        if len(wu.word) == 0:
            continue
        jj = torch.tensor([inv[w.lower()] for w in wu.word], device="cuda", dtype=torch.int64)
        pp = torch.from_numpy(pids.astype(np.int64)).cuda()[
            torch.from_numpy(wu.row.astype(np.int64)).cuda()]
        q = pp * nV + jj
        pos = torch.searchsorted(key_t, q).clamp(max=len(key_t) - 1)
        hit = key_t[pos] == q
        if not hit.any():
            continue
        st = wu.states[hit].cpu().numpy()
        js = jj[hit].cpu().numpy()
        sl = slot_t[pos[hit]].cpu().numpy()
        arr[js, sl] = st
        if s % (bs * 100) == 0:
            print(f"shard {shard}: {s}/{len(order)} {(s+bs)/max(time.time()-t0,1e-9):.0f}/s",
                  flush=True)
    arr.flush()
    print(f"shard {shard} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=16)
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--layers", required=True, help="comma-separated hidden-layer indices")
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--per-gpu", type=int, default=2)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    if a.shard >= 0:
        worker(a.shard, a.n_shards, a.encoder, layers, a.bs)
    else:
        p = store_path(a.encoder, layers)
        if not p.exists():
            z = np.load(paths.ART / "vocab.npz")
            nV, K = z["occ_pid"].shape
            np.lib.format.open_memmap(p, mode="w+", dtype=np.float16,
                                      shape=(nV, K, len(layers), 768))
            lg.info(f"allocated {p}")
        with Timer(f"per-occurrence states {a.encoder} L{layers}", lg):
            gpu_map(os.path.abspath(__file__), a.n_shards,
                    ["--encoder", a.encoder, "--layers", a.layers, "--bs", str(a.bs)],
                    logger=lg, per_gpu=a.per_gpu)
