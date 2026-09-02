"""SPLADE++ expansion terms for the related-term set (§A.3) and the SPLADE overlap (§A.2.8)."""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import Collection, load_splits
from dvlsr.encoders import SpladeEncoder
from dvlsr.util import get_logger, rng, Timer

lg = get_logger("splade-exp", "16_splade_expansions.log")
TOPK = 50


def main(force=False):
    out = paths.ART / "splade_expansions.npz"
    if out.exists() and not force:
        lg.info("exists"); return
    col = Collection()
    Q = load_splits()["Q"]
    g = rng("splade-exp")
    overlap_pids = np.sort(g.choice(Q, size=1000, replace=False))
    probes = np.load(paths.ART / "probe_plan.npz")["probes"]
    slice_id = np.load(paths.ART / "probe_plan.npz")["slice_id"]
    rep_pids = np.unique(probes[slice_id == 0][:, 0])
    rel_pids = np.sort(g.choice(rep_pids, size=min(5000, len(rep_pids)), replace=False))
    pids = np.unique(np.concatenate([overlap_pids, rel_pids]))

    sp = SpladeEncoder()
    idx = np.zeros((len(pids), TOPK), np.int32)
    val = np.zeros((len(pids), TOPK), np.float32)
    B = 128
    with Timer(f"SPLADE expansions for {len(pids)} passages", lg):
        for s in range(0, len(pids), B):
            reps = sp.encode(col.texts(pids[s:s + B]))
            for k, (ii, vv) in enumerate(reps):
                o = np.argsort(-vv)[:TOPK]
                idx[s + k, : len(o)] = ii[o]
                val[s + k, : len(o)] = vv[o]
    np.savez(out, pids=pids, idx=idx, val=val,
             vocab=np.asarray(sp.vocab), overlap_pids=overlap_pids, rel_pids=rel_pids)
    lg.info(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
