"""Build the retrieval sub-corpus C1 (§0.2).

Union of: dev-small qrels passages, BM25 top-100, SPLADE++ top-100, the dense
candidates' top-100 (both candidates -- see notes/deviations.md D2), and 500k
random passages. Every reference system's own candidates are included, which is
what makes the comparison fair; ours cannot be (the method does not exist yet),
so C1 stays mildly optimistic for us.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import load_qrels
from dvlsr.util import get_logger, rng, save_json

lg = get_logger("c1", "05_build_c1.log")


def main(force=False):
    out = paths.PREP / "c1_pids.npy"
    if out.exists() and not force:
        lg.info("C1 exists"); return
    parts, sizes = [], {}
    _, qd = load_qrels(paths.QRELS_DEV_SMALL)
    parts.append(np.unique(qd).astype(np.int32)); sizes["qrels"] = len(parts[-1])
    for name in ("bm25_full_top100", "spladepp_full_top100"):
        z = np.load(paths.RUNS / f"{name}.npz")
        d = z["docs"].reshape(-1); d = d[d >= 0]
        parts.append(np.unique(d).astype(np.int32)); sizes[name] = len(parts[-1])
    for enc in ("e5", "bge"):
        p = paths.RUNS / f"dense_{enc}_full_top1000.npz"
        if not p.exists():
            lg.warning(f"missing {p}; skipping"); continue
        d = np.load(p)["docs"][:, :100].reshape(-1)
        parts.append(np.unique(d[d >= 0]).astype(np.int32)); sizes[f"dense_{enc}"] = len(parts[-1])
    g = rng("c1-random")
    parts.append(g.choice(paths.N_PASSAGES, size=paths.N_RANDOM_C1,
                          replace=False).astype(np.int32))
    sizes["random"] = paths.N_RANDOM_C1
    pids = np.unique(np.concatenate(parts))
    np.save(out, pids)
    sizes["union"] = len(pids)
    save_json(sizes, paths.RESULTS / "05_c1.json")
    lg.info(f"C1: {sizes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
