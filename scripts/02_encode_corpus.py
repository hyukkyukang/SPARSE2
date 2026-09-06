"""Full-collection dense encoding (both candidate encoders).

Needed for (a) the §Appendix sanity check that our pipeline reproduces the dense
encoder's published MRR@10 and (b) the dense candidates that go into C1 (§0.2).
Each shard writes straight into a preallocated memmap.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import Collection
from dvlsr.encoders import Encoder
from dvlsr.util import get_logger, gpu_map

lg = get_logger("enc-corpus", "02_encode_corpus.log")


def out_path(enc: str):
    return paths.EMB / f"corpus_{enc}.npy"


def worker(shard: int, n_shards: int, enc_name: str, bs: int, subset: str = ""):
    col = Collection()
    if subset == "c1":                      # only the C1 rows of the full-size memmap
        allp = np.load(paths.PREP / "c1_pids.npy")
    else:
        allp = np.arange(len(col))
    n = len(allp)
    bounds = np.linspace(0, n, n_shards + 1).astype(np.int64)
    lo, hi = int(bounds[shard]), int(bounds[shard + 1])
    arr = np.lib.format.open_memmap(out_path(enc_name), mode="r+")
    enc = Encoder(enc_name)
    idx = allp[lo:hi]
    lens = np.asarray([col.off[i + 1] - col.off[i] for i in idx])
    order = np.argsort(lens)                      # length-sorted batches: less padding
    t0 = time.time()
    for s in range(0, len(order), bs):
        sel = idx[order[s:s + bs]]
        vecs = enc.encode_pooled(col.texts(sel), is_query=False, batch_size=bs)
        arr[sel] = vecs
        if s % (bs * 200) == 0:
            done = s + bs
            print(f"shard {shard}: {done}/{len(order)} {done/max(time.time()-t0,1e-9):.0f}/s",
                  flush=True)
    arr.flush()
    print(f"shard {shard} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=16)
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--subset", default="", choices=["", "c1"])
    a = ap.parse_args()
    if a.shard >= 0:
        worker(a.shard, a.n_shards, a.encoder, a.bs, a.subset)
    else:
        p = out_path(a.encoder)
        if not p.exists():
            np.lib.format.open_memmap(p, mode="w+", dtype=np.float16,
                                      shape=(paths.N_PASSAGES, 768))
            lg.info(f"allocated {p}")
        gpu_map(os.path.abspath(__file__), a.n_shards,
                ["--encoder", a.encoder, "--bs", str(a.bs)]
                + (["--subset", a.subset] if a.subset else []), logger=lg, per_gpu=a.per_gpu)
        lg.info(f"corpus encoded: {p} (subset={a.subset or 'full'})")
