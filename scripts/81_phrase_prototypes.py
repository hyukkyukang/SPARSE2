"""Phrase / entity insertion pilot, step 2: contextual prototypes for the phrase entries.

A phrase occurrence's state is the mean of its two word units' states (§0.4 extended
to a multi-word unit); the prototype is the mean over the sampled occurrences, at
every layer, in two disjoint halves exactly as scripts/11_prototypes.py does for
words. Output: artifacts/proto_{enc}_phr.npz = word prototypes + phrase prototypes,
same layout as proto_{enc}.npz so every downstream script can take --vocab-tag phr.
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

lg = get_logger("phrproto", "81_phrase_prototypes.log")
K_HALF = paths.PROTO_K


def phrase_rows():
    z = np.load(paths.vocab_file("phr"))
    isp = z["is_phrase"]
    words = [str(w) for w in z["words"]]
    rows = np.flatnonzero(isp)
    return z, rows, [words[i] for i in rows]


def worker(shard, n_shards, enc_name, bs):
    z, rows, phrases = phrase_rows()
    occ = z["occ_pid"][rows]                             # (n_phr, 100)
    n_phr = len(rows)
    inv = {p: i for i, p in enumerate(phrases)}
    # (pid, phrase) pairs that were sampled, with their half
    j = np.repeat(np.arange(n_phr, dtype=np.int64), occ.shape[1])
    pid = occ.reshape(-1).astype(np.int64)
    half = np.tile((np.arange(occ.shape[1]) >= K_HALF).astype(np.int64), n_phr)
    ok = pid >= 0
    key = pid[ok] * n_phr + j[ok]
    half = half[ok]
    o = np.argsort(key)
    key, half = key[o], half[o]
    upid = np.unique(pid[ok])
    bounds = np.linspace(0, len(upid), n_shards + 1).astype(np.int64)
    mine = upid[bounds[shard]:bounds[shard + 1]]
    L = paths.layers_for(enc_name)
    col = Collection()
    enc = get_encoder(enc_name)
    dev = "cuda"
    sums = torch.zeros(2, n_phr, len(L), enc.dim, device=dev, dtype=torch.float32)
    cnts = torch.zeros(2, n_phr, device=dev, dtype=torch.float32)
    key_t = torch.from_numpy(key).to(dev)
    half_t = torch.from_numpy(half).to(dev)
    lens = np.asarray([col.off[i + 1] - col.off[i] for i in mine])
    order = np.argsort(lens)
    t0 = time.time()
    for s in range(0, len(order), bs):
        pids = mine[order[s:s + bs]]
        wu = enc.encode_word_units(col.texts(pids), L, is_query=False)
        if len(wu.word) < 2:
            continue
        # consecutive units within the same row whose lowercase pair is a phrase
        row = wu.row
        a_idx, jj = [], []
        for u in range(len(wu.word) - 1):
            if row[u] != row[u + 1]:
                continue
            k = wu.word[u].lower() + " " + wu.word[u + 1].lower()
            pj = inv.get(k)
            if pj is not None:
                a_idx.append(u); jj.append(pj)
        if not a_idx:
            continue
        a_idx = torch.as_tensor(a_idx, device=dev)
        jj = torch.as_tensor(jj, device=dev)
        pp = torch.from_numpy(pids.astype(np.int64)).to(dev)[
            torch.from_numpy(row.astype(np.int64)).to(dev)[a_idx]]
        q = pp * n_phr + jj
        pos = torch.searchsorted(key_t, q).clamp(max=len(key_t) - 1)
        hit = key_t[pos] == q
        if not hit.any():
            continue
        h = half_t[pos[hit]]
        st = 0.5 * (wu.states[a_idx[hit]].float() + wu.states[a_idx[hit] + 1].float())
        tgt = h * n_phr + jj[hit]
        sums.view(2 * n_phr, len(L), enc.dim).index_add_(0, tgt, st)
        cnts.view(2 * n_phr).index_add_(0, tgt, torch.ones(len(tgt), device=dev))
        if s % (bs * 50) == 0:
            print(f"shard {shard}: {s}/{len(order)} "
                  f"{(s+bs)/max(time.time()-t0,1e-9):.0f}/s", flush=True)
    np.savez(paths.ART / f"phrproto_part_{enc_name}_{shard}.npz",
             sums=sums.cpu().numpy(), cnts=cnts.cpu().numpy())
    print(f"shard {shard} done {time.time()-t0:.0f}s", flush=True)


def merge(enc_name, n_shards):
    S = C = None
    for s in range(n_shards):
        z = np.load(paths.ART / f"phrproto_part_{enc_name}_{s}.npz")
        S = z["sums"] if S is None else S + z["sums"]
        C = z["cnts"] if C is None else C + z["cnts"]
    proto = S / np.maximum(C, 1)[:, :, None, None]
    base = np.load(paths.proto_file(enc_name))
    np.savez(paths.proto_file(enc_name, "phr"),
             protoA=np.concatenate([base["protoA"], proto[0].astype(np.float16)]),
             protoB=np.concatenate([base["protoB"], proto[1].astype(np.float16)]),
             cntA=np.concatenate([base["cntA"], C[0]]),
             cntB=np.concatenate([base["cntB"], C[1]]),
             layers=base["layers"])
    lg.info(f"{enc_name}: phrase prototypes {proto.shape[1:]} "
            f"countA min={C[0].min():.0f} med={np.median(C[0]):.0f} "
            f"countB min={C[1].min():.0f} med={np.median(C[1]):.0f}; "
            f"phrases with countA==0: {int((C[0] == 0).sum())}")
    for s in range(n_shards):
        os.remove(paths.ART / f"phrproto_part_{enc_name}_{s}.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=10)
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--per-gpu", type=int, default=2)
    a = ap.parse_args()
    if a.shard >= 0:
        worker(a.shard, a.n_shards, a.encoder, a.bs)
    else:
        with Timer(f"phrase prototypes {a.encoder}", lg):
            gpu_map(os.path.abspath(__file__), a.n_shards,
                    ["--encoder", a.encoder, "--bs", str(a.bs)], logger=lg,
                    per_gpu=a.per_gpu)
        merge(a.encoder, a.n_shards)
