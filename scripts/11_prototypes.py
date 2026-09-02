"""R2 contextual prototypes (§A.1): v_j = mean word-unit state over k occurrences of P.

Streaming, as §A.2 requires: individual occurrence states are never stored, only
the running per-(entry, layer) sums. Two disjoint halves (slots 0-49 and 50-99)
are accumulated so Pilot B's stability curve has disjoint occurrence sets.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import Collection
from dvlsr.encoders import get_encoder
from dvlsr.util import get_logger, gpu_map, Timer

lg = get_logger("proto", "11_prototypes.log")
K_HALF = paths.PROTO_K          # 50


def pair_keys():
    z = np.load(paths.ART / "vocab.npz")
    occ = z["occ_pid"]                                # (|V|, 100)
    nV = occ.shape[0]
    j = np.repeat(np.arange(nV, dtype=np.int64), occ.shape[1])
    pid = occ.reshape(-1).astype(np.int64)
    half = np.tile((np.arange(occ.shape[1]) >= K_HALF).astype(np.int8), nV)
    ok = pid >= 0
    key = pid[ok] * nV + j[ok]
    half = half[ok]
    o = np.argsort(key)
    return key[o], half[o], nV, np.unique(pid[ok]).astype(np.int64)


def worker(shard, n_shards, enc_name, bs):
    key, half, nV, upid = pair_keys()
    bounds = np.linspace(0, len(upid), n_shards + 1).astype(np.int64)
    mine = upid[bounds[shard]:bounds[shard + 1]]
    z = np.load(paths.ART / "vocab.npz")
    words = [str(w) for w in z["words"]]
    inv = {w: i for i, w in enumerate(words)}
    L = paths.layers_for(enc_name)
    col = Collection()
    enc = get_encoder(enc_name)
    dev = "cuda"
    sums = torch.zeros(2, nV, len(L), enc.dim, device=dev, dtype=torch.float32)
    cnts = torch.zeros(2, nV, device=dev, dtype=torch.float32)
    key_t = torch.from_numpy(key).to(dev)
    half_t = torch.from_numpy(half.astype(np.int64)).to(dev)

    lens = np.asarray([col.off[i + 1] - col.off[i] for i in mine])
    order = np.argsort(lens)
    t0 = time.time()
    for s in range(0, len(order), bs):
        pids = mine[order[s:s + bs]]
        wu = enc.encode_word_units(col.texts(pids), L, is_query=False,
                                   keep=lambda w: w in inv)
        if len(wu.word) == 0:
            continue
        jj = torch.tensor([inv[w.lower()] for w in wu.word], device=dev, dtype=torch.int64)
        pp = torch.from_numpy(pids.astype(np.int64)).to(dev)[
            torch.from_numpy(wu.row.astype(np.int64)).to(dev)]
        q = pp * nV + jj
        pos = torch.searchsorted(key_t, q)
        pos = pos.clamp(max=len(key_t) - 1)
        hit = key_t[pos] == q
        if not hit.any():
            continue
        h = half_t[pos[hit]]
        tgt = h * nV + jj[hit]
        st = wu.states[hit].float()
        sums.view(2 * nV, len(L), enc.dim).index_add_(0, tgt, st)
        cnts.view(2 * nV).index_add_(0, tgt, torch.ones(len(tgt), device=dev))
        if s % (bs * 100) == 0:
            print(f"shard {shard}: {s}/{len(order)} {(s+bs)/max(time.time()-t0,1e-9):.0f}/s",
                  flush=True)
    np.savez(paths.ART / f"proto_part_{enc_name}_{shard}.npz",
             sums=sums.cpu().numpy().astype(np.float32), cnts=cnts.cpu().numpy())
    print(f"shard {shard} done {time.time()-t0:.0f}s", flush=True)


def merge(enc_name, n_shards):
    S = C = None
    for s in range(n_shards):
        z = np.load(paths.ART / f"proto_part_{enc_name}_{s}.npz")
        S = z["sums"] if S is None else S + z["sums"]
        C = z["cnts"] if C is None else C + z["cnts"]
    proto = S / np.maximum(C, 1)[:, :, None, None]
    np.savez(paths.ART / f"proto_{enc_name}.npz",
             protoA=proto[0].astype(np.float16), protoB=proto[1].astype(np.float16),
             cntA=C[0], cntB=C[1], layers=np.asarray(paths.layers_for(enc_name)))
    lg.info(f"{enc_name}: prototypes {proto.shape[1:]} "
            f"countA min={C[0].min():.0f} med={np.median(C[0]):.0f} "
            f"countB min={C[1].min():.0f} med={np.median(C[1]):.0f}")
    for s in range(n_shards):
        os.remove(paths.ART / f"proto_part_{enc_name}_{s}.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=16)
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--per-gpu", type=int, default=2)
    a = ap.parse_args()
    if a.shard >= 0:
        worker(a.shard, a.n_shards, a.encoder, a.bs)
    else:
        with Timer(f"prototypes {a.encoder}", lg):
            gpu_map(os.path.abspath(__file__), a.n_shards,
                    ["--encoder", a.encoder, "--bs", str(a.bs)], logger=lg,
                    per_gpu=a.per_gpu)
        merge(a.encoder, a.n_shards)
