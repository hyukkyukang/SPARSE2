"""§0.5 — estimate every whitening transform once and freeze it.

Token side: W_H/mu_H from B_H, W_Q/mu_Q from B_Q (per encoder, per layer).
Entry side: one transform per representation from that representation's own matrix.
Also runs the §0.5 check on whether the query side needs its own transform.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("whiten", "13_whitening.log")
WD = paths.ART / "whiten"
WD.mkdir(exist_ok=True)


def _cov(X):
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    Xc = X - mu
    return mu, (Xc.T @ Xc) / (len(X) - 1)


def main(enc_name):
    L = paths.layers_for(enc_name)
    bH = np.load(paths.ART / f"bankH_{enc_name}.npy")        # (n, |L|, d)
    bQ = np.load(paths.ART / f"bankQ_{enc_name}.npy")
    proto = np.load(paths.ART / f"proto_{enc_name}.npz")["protoA"]
    r1 = np.load(paths.ART / f"r1_{enc_name}.npz")
    stats = {}
    for li, l in enumerate(L):
        for tag, X in (("H", bH[:, li]), ("Q", bQ[:, li]), ("V_R2", proto[:, li])):
            mu, W, eps = whitening.estimate(X.astype(np.float32))
            whitening.save(WD / f"{enc_name}_L{l}_{tag}.npz", mu, W, eps)
        muH, covH = _cov(bH[:, li][:50000])
        muQ, covQ = _cov(bQ[:, li][:50000])
        dmu, dcov = whitening.cov_compare(muQ, covQ, muH, covH)
        stats[f"L{l}"] = dict(rel_mu_shift=dmu, rel_cov_frob=dcov,
                              shared_ok=bool(dmu < 0.05 and dcov < 0.05))
        lg.info(f"{enc_name} L{l}: ||muQ-muH||/||muH|| = {dmu:.3f}  covFrob = {dcov:.3f}")
    for tag in r1.files:
        mu, W, eps = whitening.estimate(r1[tag].astype(np.float32))
        whitening.save(WD / f"{enc_name}_R1{tag}_V.npz", mu, W, eps)
    save_json(stats, paths.RESULTS / f"13_whiten_qshift_{enc_name}.json")
    lg.info(f"{enc_name}: whitening transforms frozen in {WD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder", default="e5")
    with Timer("whitening", lg):
        main(ap.parse_args().encoder)
