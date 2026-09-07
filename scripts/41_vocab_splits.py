"""§D.4 — the three held-out vocabulary splits.

random-stratified: 20% within each frequency decile (stopwords never held out)
cluster:           k-means (k=100) over whitened entry vectors, whole clusters held out
rare:              the entire bottom frequency decile
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.util import get_logger, rng, save_json

lg = get_logger("splits", "41_vocab_splits.log")


def entry_vectors(enc, layer, rep, r1_prefix="none"):
    if rep == "R2":
        return np.asarray(np.load(paths.ART / f"proto_{enc}.npz")["protoA"][
            :, paths.layers_for(enc).index(layer)], np.float32)
    return np.asarray(np.load(paths.ART / f"r1_{enc}.npz")[r1_prefix], np.float32)


def main(enc, layer, rep, r1_prefix):
    z = np.load(paths.ART / "vocab.npz")
    decile, is_stop, freq = z["decile"], z["is_stop"], z["freq"]
    words = [str(w) for w in z["words"]]
    nV = len(words)
    out = {}

    g = rng("split-random")
    held = np.zeros(nV, bool)
    for d in range(10):
        cand = np.flatnonzero((decile == d) & ~is_stop)
        k = int(round(0.2 * len(cand)))
        held[g.choice(cand, size=k, replace=False)] = True
    out["random"] = held.copy()

    X = entry_vectors(enc, layer, rep, r1_prefix)
    mu, W, _ = whitening.estimate(X)
    Xw = (X - mu) @ W
    Xw /= np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-9
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=100, n_init=4, random_state=0).fit(Xw)
    lab = km.labels_
    g = rng("split-cluster")
    order = g.permutation(100)
    held = np.zeros(nV, bool)
    for c in order:
        cand = (lab == c) & ~is_stop
        if held.sum() + cand.sum() > 0.25 * nV:
            continue
        held |= cand
        if held.sum() >= 0.15 * nV:
            break
    out["cluster"] = held.copy()

    held = (decile == 9) & ~is_stop
    out["rare"] = held.copy()

    stats = {}
    for k, h in out.items():
        stats[k] = dict(n_held=int(h.sum()), frac=float(h.mean()),
                        by_decile={int(d): int((h & (decile == d)).sum()) for d in range(10)},
                        example_words=[words[i] for i in np.flatnonzero(h)[:15]])
        lg.info(f"{k}: {h.sum()} held ({h.mean():.3f}) "
                f"deciles={ {d: int((h & (decile==d)).sum()) for d in range(10)} }")
    np.savez(paths.PREP / "vocab_splits.npz", **{k: v for k, v in out.items()},
             cluster_labels=lab)
    save_json(stats, paths.RESULTS / "41_vocab_splits.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--rep", default="R2")
    ap.add_argument("--r1-prefix", default="none")
    a = ap.parse_args()
    main(a.encoder, a.layer, a.rep, a.r1_prefix)
