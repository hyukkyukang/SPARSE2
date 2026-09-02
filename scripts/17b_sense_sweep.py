"""Recompute sense accuracy for every Pilot A configuration and patch the results.

Split out from 14_pilotA.py because the sense metric depends only on the 2,400
polysemy occurrences and the entry matrix: recomputing it after the labels were
corrected costs seconds per configuration instead of a full re-sweep.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.probes import sense_accuracy
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("sense", "17b_sense_sweep.log")
WD = paths.ART / "whiten"
DEV = "cuda"


def make_tf(mode, path):
    if mode == "raw":
        return whitening.Transform("raw")
    mu, W = whitening.load(path)
    return whitening.Transform(mode, mu, W, DEV)


def entry_matrix(rep, enc, li, layer, pfx, shared):
    if rep == "R2":
        E = np.load(paths.ART / f"proto_{enc}.npz")["protoA"][:, li]
        wp = WD / f"{enc}_L{layer}_V_R2.npz"
    else:
        E = np.load(paths.ART / f"r1_{enc}.npz")[pfx]
        wp = WD / f"{enc}_R1{pfx}_V.npz"
    if shared:
        wp = WD / f"{enc}_L{layer}_H.npz"
    return np.asarray(E, np.float32), wp


def main(enc, r1_prefix):
    aux = dict(np.load(paths.ART / f"aux2_{enc}.npz", allow_pickle=True))
    poly = dict(np.load(paths.ART / "polysemy.npz", allow_pickle=True))
    poly["filled"] = aux["poly_filled"]
    L = paths.layers_for(enc)
    res_path = paths.RESULTS / f"14_pilotA_{enc}_report.json"
    res = json.load(open(res_path)) if res_path.exists() else {}
    out = {}
    for li, layer in enumerate(L):
        for tr in ("raw", "centered", "whitened"):
            tfH = make_tf(tr, WD / f"{enc}_L{layer}_H.npz")
            for rep in ("R1", "R2"):
                shareds = [False, True] if rep == "R1" else [False]
                for shared in shareds:
                    if shared and tr != "whitened":
                        continue
                    pfx = r1_prefix if rep == "R1" else None
                    E_np, wp = entry_matrix(rep, enc, li, layer, pfx, shared)
                    E = make_tf(tr, wp)(torch.as_tensor(E_np, device=DEV)).half()
                    s = sense_accuracy(aux["poly_states"][:, li], poly, tfH, E)
                    key = (f"L{layer}|{rep}" + (f"-{pfx}" if pfx else "")
                           + ("-shared" if shared else "") + f"|{tr}")
                    out[key] = s
                    if key in res:
                        res[key]["sense_v1"] = res[key].get("sense")
                        res[key]["sense"] = s
                    del E
                    torch.cuda.empty_cache()
            lg.info(f"{enc} L{layer} {tr}: " + " ".join(
                f"{k.split('|')[1]}={v['accuracy']:.3f}"
                for k, v in out.items() if k.startswith(f"L{layer}|") and k.endswith(tr)))
    if res:
        save_json(res, res_path)
    save_json(out, paths.RESULTS / f"17b_sense_{enc}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--r1-prefix", default="none")
    a = ap.parse_args()
    with Timer(f"sense sweep {a.encoder}", lg):
        main(a.encoder, a.r1_prefix)
