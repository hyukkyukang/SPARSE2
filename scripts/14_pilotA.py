"""Pilot A — token-state geometry against a text-defined vocabulary (§A).

Sweeps (encoder x layer x transform x entry representation) and streams every
similarity profile: nothing of size |probes| x |V| is ever stored (§0.7).
"""
from __future__ import annotations
import argparse, functools, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.data import Collection
from dvlsr.encoders import get_encoder
from dvlsr.sparse import profile_maxpool, find_tau
from dvlsr.util import get_logger, save_json, rng, Timer

lg = get_logger("pilotA", "14_pilotA.log")
WD = paths.ART / "whiten"
DEV = "cuda"
LAYERS_CUR = list(paths.LAYERS)
CHUNK = 4096


# ----------------------------------------------------------------- aux states
def aux_states(enc_name, force=False):
    """Word-unit states for the SPLADE-overlap passages and the polysemy occurrences."""
    p = paths.ART / f"aux2_{enc_name}.npz"
    if p.exists() and not force:
        return dict(np.load(p, allow_pickle=True))
    col = Collection()
    exp = np.load(paths.ART / "splade_expansions.npz", allow_pickle=True)
    poly = np.load(paths.ART / "polysemy.npz", allow_pickle=True)
    enc = get_encoder(enc_name)
    L = paths.layers_for(enc_name)
    out = {}

    ov = exp["overlap_pids"]
    rows, st = [], []
    for s in range(0, len(ov), 128):
        wu = enc.encode_word_units(col.texts(ov[s:s + 128]), L)
        rows.append(wu.row.astype(np.int32) + s); st.append(wu.states.cpu().numpy())
    out["ov_row"] = np.concatenate(rows); out["ov_states"] = np.concatenate(st, 0)
    out["ov_pids"] = ov

    z = np.load(paths.ART / "vocab.npz")
    inv = {str(w): i for i, w in enumerate(z["words"])}
    pw = [str(x) for x in poly["words"]]
    ppid = poly["pids"]
    states = np.zeros((len(ppid), len(L), enc.dim), np.float16)
    filled = np.zeros(len(ppid), bool)
    for s in range(0, len(ppid), 128):
        sl = slice(s, s + 128)
        wu = enc.encode_word_units(col.texts(ppid[sl]), L,
                                   keep=lambda w: w in inv)
        seen = {}
        for u in range(len(wu.word)):
            key = (int(wu.row[u]), wu.word[u].lower())
            seen.setdefault(key, u)
        for k, i in enumerate(range(s, min(s + 128, len(ppid)))):
            u = seen.get((k, pw[i]))
            if u is not None:
                states[i] = wu.states[u].cpu().numpy(); filled[i] = True
    out["poly_states"] = states; out["poly_filled"] = filled
    np.savez(p, **out)
    lg.info(f"{enc_name}: aux2 saved (overlap units {out['ov_states'].shape}, "
            f"polysemy filled {filled.mean():.3f})")
    return dict(np.load(p, allow_pickle=True))


# ----------------------------------------------------------------- transforms
def make_tf(mode, side_path):
    if mode == "raw":
        return whitening.Transform("raw")
    mu, W = whitening.load(side_path)
    return whitening.Transform(mode, mu, W, DEV)


@functools.lru_cache(maxsize=4)
def _proto(enc_name):
    return np.load(paths.ART / f"proto_{enc_name}.npz")["protoA"]


@functools.lru_cache(maxsize=4)
def _r1(enc_name):
    return dict(np.load(paths.ART / f"r1_{enc_name}.npz"))


def entry_matrix(rep, enc_name, li, r1_prefix, r1_shared=False):
    if rep == "R2":
        E = _proto(enc_name)[:, li]
        wpath = WD / f"{enc_name}_L{LAYERS_CUR[li]}_V_R2.npz"
    else:
        E = _r1(enc_name)[r1_prefix]
        wpath = WD / f"{enc_name}_R1{r1_prefix}_V.npz"
    if r1_shared:
        wpath = WD / f"{enc_name}_L{LAYERS_CUR[li]}_H.npz"
    return np.asarray(E, np.float32), wpath


# ----------------------------------------------------------------- metrics
from dvlsr.probes import (profile_chunks, probe_metrics, summarize,
                          sense_accuracy, splade_overlap)


# ----------------------------------------------------------------- main sweep
def _related_tensors(probes, idx, exp, inv, n_terms=10, seed=7):
    """Padded (related-term, mask, matched-random) tensors for the reporting probes."""
    pid2e = {int(p): i for i, p in enumerate(exp["pids"])}
    sp_vocab = [str(x) for x in exp["vocab"]]
    T = np.full((len(idx), n_terms), -1, np.int64)
    for k, i in enumerate(idx):
        pid, j = probes[i]
        e = pid2e.get(int(pid))
        if e is None:
            continue
        c = 0
        for t in exp["idx"][e]:
            w = inv.get(sp_vocab[t])
            if w is not None and w != int(j):
                T[k, c] = w
                c += 1
                if c == n_terms:
                    break
    M = T >= 0
    g = rng("related-random", seed)
    R = g.integers(0, len(inv), size=T.shape).astype(np.int64)
    return (torch.as_tensor(T, device=DEV), torch.as_tensor(M, device=DEV),
            torch.as_tensor(R, device=DEV))


def run(enc_name, layers=None, reps=("R1", "R2"),
        transforms=("raw", "centered", "whitened"), do_sense=True, do_overlap=True,
        r1_prefix=None, mode="report"):
    z = np.load(paths.ART / "vocab.npz")
    words = [str(w) for w in z["words"]]
    inv = {w: i for i, w in enumerate(words)}
    is_stop_v, decile_v = z["is_stop"], z["decile"]
    pr = np.load(paths.ART / f"probes_{enc_name}.npz", allow_pickle=True)
    probes, slice_id, filled = pr["probes"], pr["slice_id"], pr["filled"]
    global LAYERS_CUR
    L = list(paths.layers_for(enc_name))
    LAYERS_CUR = L
    layers = layers or L
    tauS = np.load(paths.ART / f"tauS_{enc_name}.npz", allow_pickle=True)
    exp = np.load(paths.ART / "splade_expansions.npz", allow_pickle=True)
    poly = dict(np.load(paths.ART / "polysemy.npz", allow_pickle=True))
    aux = aux_states(enc_name)
    poly["filled"] = aux["poly_filled"]

    keep = ((slice_id == 1) if mode == "prefix" else (slice_id == 0)) & filled
    idx = np.flatnonzero(keep)
    own = torch.as_tensor(probes[idx, 1], device=DEV, dtype=torch.long)
    pstop, pdec = is_stop_v[probes[idx, 1]], decile_v[probes[idx, 1]]
    T, M, R = _related_tensors(probes, idx, exp, inv)
    rowS = torch.as_tensor(tauS["row"].astype(np.int64), device=DEV)
    n_rowS = int(tauS["n_rows"])
    lg.info(f"{enc_name} [{mode}]: {len(idx)} probes, "
            f"{int((M.sum(1) > 0).sum())} with related terms")

    results = {}
    for li, l in [(L.index(x), x) for x in layers]:
        probe_raw = torch.as_tensor(pr["states"][idx, li], device=DEV)
        tauS_raw = torch.as_tensor(tauS["states"][:, li], device=DEV)
        for tr in transforms:
            tfH = make_tf(tr, WD / f"{enc_name}_L{l}_H.npz")
            H = tfH(probe_raw).half()
            Hs = tfH(tauS_raw).half()
            for rep in reps:
                if rep == "R1":
                    pfxs = (list(np.load(paths.ART / f"r1_{enc_name}.npz").files)
                            if mode == "prefix" else [r1_prefix or "doc"])
                    shareds = [False] if mode == "prefix" else [False, True]
                else:
                    pfxs, shareds = [None], [False]
                for pfx in pfxs:
                    for shared in shareds:
                        if shared and tr != "whitened":
                            continue          # only the whitened space has a shared transform
                        E_np, wpath = entry_matrix(rep, enc_name, li, pfx, shared)
                        E = make_tf(tr, wpath)(torch.as_tensor(E_np, device=DEV)).half()
                        PS = profile_maxpool(Hs, rowS, n_rowS, E)
                        tau = find_tau(PS, paths.NNZ_DOC_TARGET)
                        tok_pct = float((PS.reshape(-1) < tau).float().mean() * 100)
                        del PS
                        m = probe_metrics(H, own, E, tau, T, M, R)
                        r = summarize(m, pstop, pdec)
                        r.update(tau=tau, tau_token_percentile=tok_pct)
                        if do_sense:
                            r["sense"] = sense_accuracy(aux["poly_states"][:, li],
                                                        poly, tfH, E)
                        if do_overlap:
                            r["splade_overlap"] = splade_overlap(
                                aux, exp, tfH, E, tau, words, inv, exp["vocab"], li)
                        key = (f"L{l}|{rep}" + (f"-{pfx}" if pfx else "")
                               + ("-shared" if shared else "") + f"|{tr}")
                        results[key] = r
                        lg.info(
                            f"{enc_name} {key}: hit10={r['content']['self_hit10']:.3f} "
                            f"(stop {r['stop']['self_hit10']:.3f}) "
                            f"relMRR={r.get('related_mrr', 0):.4f}"
                            f"/{r.get('related_ratio', 0):.1f}x "
                            f"z={r['content']['zgap']:.2f} "
                            f"PR={r['content']['pr']:.1f} "
                            f"nnz/tok={r['content']['nnz_tok']:.1f}"
                            + (f" sense={r['sense']['accuracy']:.3f}" if do_sense else "")
                            + (f" jac={r['splade_overlap']['jaccard']:.3f}"
                               f"/cov={r['splade_overlap']['coverage']:.2f}"
                               if do_overlap else ""))
                        del E
                        torch.cuda.empty_cache()
            del H, Hs
        del probe_raw, tauS_raw
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--mode", default="report", choices=["report", "prefix"])
    ap.add_argument("--layers", default="")
    ap.add_argument("--reps", default="R1,R2")
    ap.add_argument("--r1-prefix", default=None)
    ap.add_argument("--no-sense", action="store_true")
    ap.add_argument("--no-overlap", action="store_true")
    a = ap.parse_args()
    ls = [int(x) for x in a.layers.split(",")] if a.layers else None
    with Timer(f"pilot A {a.encoder} [{a.mode}]", lg):
        res = run(a.encoder, ls, tuple(a.reps.split(",")), do_sense=not a.no_sense,
                  do_overlap=not a.no_overlap, r1_prefix=a.r1_prefix, mode=a.mode)
    out = paths.RESULTS / f"14_pilotA_{a.encoder}_{a.mode}.json"
    save_json(res, out)
    lg.info(f"saved {out}")
