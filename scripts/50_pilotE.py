"""Pilot E (conditional) — insertion-time calibration (§E).

Every variant is a per-entry *affine* map on the logits, applied before the
threshold, so each is one (scale, shift) vector and each is training-free and
computable at insertion from statistics a new entry already has.

  Z  : match the entry's background mean/std on B_H to the seen median of its decile
  DF : shift the entry's threshold so its df on S matches the seen median of its decile
  SA : scale so its self-activation on its own word matches the seen median
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.data import Collection, load_splits
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("pilotE", "50_pilotE.log")
DEV = "cuda"


def logits_over(bank, head, E, tf, chunk=8192):
    """Per-entry mean and std of z over a set of token states."""
    n = bank.shape[0]
    s1 = torch.zeros(E.shape[0], device=DEV, dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    for s in range(0, n, chunk):
        h = tf(torch.as_tensor(bank[s:s + chunk], device=DEV), normalize=False)
        z = head.logits(h.to(E.dtype), E).double()
        s1 += z.sum(0); s2 += (z ** 2).sum(0)
        del z
    mu = s1 / n
    sd = (s2 / n - mu ** 2).clamp(min=1e-12).sqrt()
    return mu.float().cpu().numpy(), sd.float().cpu().numpy()


def self_activation_logit(head, E, enc, tf, words, layer, bs=512):
    out = np.zeros(len(words), np.float32)
    with torch.no_grad():
        for s in range(0, len(words), bs):
            chunk = words[s:s + bs]
            wu = enc.encode_word_units(chunk, [layer])
            h = tf(wu.states[:, 0].to(DEV), normalize=False)
            z = head.logits(h.to(E.dtype), E)
            row = torch.as_tensor(wu.row.astype(np.int64), device=DEV)
            P = torch.full((len(chunk), E.shape[0]), -1e4, device=DEV, dtype=z.dtype)
            P = P.scatter_reduce(0, row.unsqueeze(1).expand_as(z), z, reduce="amax",
                                 include_self=True)
            own = torch.arange(s, s + len(chunk), device=DEV)
            out[s:s + len(chunk)] = P[torch.arange(len(chunk), device=DEV),
                                      own].float().cpu().numpy()
            del z, P
    return out


def decile_targets(vals, seen, decile):
    """Median over *seen* entries of the same frequency decile (deployable at insertion)."""
    t = np.zeros(len(vals), np.float32)
    for d in range(10):
        m = decile == d
        s = m & seen
        t[m] = np.median(vals[s]) if s.sum() else np.median(vals[seen])
    return t


def main(a):
    sys.path.insert(0, str(paths.REPO / "scripts"))
    import importlib.util as iu
    spec = iu.spec_from_file_location("e43", paths.REPO / "scripts" / "43_encode_eval.py")
    e43 = iu.module_from_spec(spec); spec.loader.exec_module(e43)
    cfg, head, enc, E, _AB = e43.load_ckpt(a.name)
    layer = cfg["layer"]
    tag = cfg.get("vocab_tag", "") or ""
    z = np.load(paths.vocab_file(tag))
    words = [str(w) for w in z["words"]]
    decile = z["decile"]
    sp = np.load(paths.vsplits_file(tag))
    held = sp[cfg["split"]]
    seen = ~held
    mu, W = whitening.load(paths.ART / "whiten" / f"{cfg['encoder']}_L{layer}_H.npz")
    tf = whitening.Transform("whitened", mu, W, DEV)
    bank = np.load(paths.ART / f"bankH_{cfg['encoder']}.npy", mmap_mode="r")
    li = paths.layers_for(cfg["encoder"]).index(layer)
    with torch.no_grad(), Timer("background logit statistics on B_H", lg):
        m_j, s_j = logits_over(np.asarray(bank[:, li]), head, E, tf)
    m_star = decile_targets(m_j, seen, decile)
    s_star = decile_targets(s_j, seen, decile)

    calibs = {}
    scale = np.ones(len(words), np.float32); shift = np.zeros(len(words), np.float32)
    calibs["raw"] = np.stack([scale, shift])
    sc = np.where(held, s_star / np.maximum(s_j, 1e-6), 1.0).astype(np.float32)
    sh = np.where(held, m_star - m_j * sc, 0.0).astype(np.float32)
    calibs["Z"] = np.stack([sc, sh])

    with torch.no_grad(), Timer("self-activation logits", lg):
        sa = self_activation_logit(head, E, enc, tf, words, layer)
    sa_star = decile_targets(sa, seen, decile)
    sc2 = np.where(held & (np.abs(sa) > 1e-6), sa_star / sa, 1.0).astype(np.float32)
    calibs["SA"] = np.stack([sc2, np.zeros_like(sc2)])

    # DF: shift so df on S matches the seen median of the decile
    si, sv = (np.load(paths.EMB / f"D_{a.name}_s_idx.npy", mmap_mode="r"),
              np.load(paths.EMB / f"D_{a.name}_s_val.npy", mmap_mode="r"))
    nS = si.shape[0]
    df = np.zeros(len(words), np.int64)
    for s in range(0, nS, 20000):
        i = np.asarray(si[s:s + 20000]); v = np.asarray(sv[s:s + 20000], np.float32)
        df += np.bincount(i[v > 0], minlength=len(words))
    df_star = decile_targets((df / nS).astype(np.float32), seen, decile)
    # a positive shift raises activation; solve monotonically by bisection on the shift
    shift_df = np.zeros(len(words), np.float32)
    lg.info("DF calibration: solving per-entry shifts by bisection on S")
    Hs = None
    for it in range(12):
        cur = np.stack([np.ones(len(words), np.float32), shift_df])
        # cheap proxy: df changes monotonically with the shift; use the stored top-k values
        est = np.zeros(len(words), np.float64)
        for s in range(0, nS, 20000):
            i = np.asarray(si[s:s + 20000]); v = np.asarray(sv[s:s + 20000], np.float32)
            zz = np.expm1(v) + shift_df[i]              # invert log1p, apply shift
            est += np.bincount(i[zz > 0], minlength=len(words))
        est /= nS
        too_low = held & (est < df_star)
        shift_df[too_low] += 0.5 / (it + 1)
        shift_df[held & (est > df_star)] -= 0.5 / (it + 1)
    calibs["DF"] = np.stack([np.ones(len(words), np.float32), shift_df])

    d = paths.ART / "calib"
    d.mkdir(exist_ok=True)
    for k, v in calibs.items():
        np.save(d / f"{a.name}_{k}.npy", v.astype(np.float32))
    save_json(dict(name=a.name, split=cfg["split"],
                   mean_scale_Z=float(calibs["Z"][0][held].mean()),
                   mean_shift_Z=float(calibs["Z"][1][held].mean()),
                   mean_scale_SA=float(calibs["SA"][0][held].mean()),
                   mean_shift_DF=float(calibs["DF"][1][held].mean())),
              paths.RESULTS / f"50_pilotE_calib_{a.name}.json")
    lg.info(f"calibration vectors written to {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    main(ap.parse_args())
