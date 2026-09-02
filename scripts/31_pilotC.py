"""Pilot C — training-free end-to-end retrieval on C1 (§C).

One end-to-end number before any training, and the sparsity operating point that
Pilot D initialises from. All 36 grid cells are re-truncations of one stored set
of profiles (§C.1).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.data import load_queries, qrels_dict, Collection
from dvlsr.metrics import evaluate, bootstrap_ci, paired_bootstrap
from dvlsr.retrieval import InvertedIndex, saturate, nnz_stats
from dvlsr.sparse import profile_maxpool, find_tau
from dvlsr.util import get_logger, save_json, rng, Timer

lg = get_logger("pilotC", "31_pilotC.log")
DEV = "cuda"
GRID_KD = [64, 128, 256]
GRID_KQ = [16, 32, 64]
GRID_SAT = ["none", "log1p"]


def store(enc, layer, rep, kind):
    b = paths.EMB / f"c1_{enc}_L{layer}_{rep}"
    return (np.load(b.with_name(b.name + f"_{kind}_idx.npy"), mmap_mode="r"),
            np.load(b.with_name(b.name + f"_{kind}_val.npy"), mmap_mode="r"))


def doc_taus(enc, layer, rep, r1_prefix, targets=(120, 60)):
    """tau_d fitted on the 5,000 S passages of §0.6, in this exact space."""
    sys.path.insert(0, str(paths.REPO / "scripts"))
    import importlib.util as iu
    spec = iu.spec_from_file_location("e30", paths.REPO / "scripts" / "30_encode_c1.py")
    m = iu.module_from_spec(spec); spec.loader.exec_module(m)
    E = m.entry_matrix(enc, layer, rep, r1_prefix)
    tf = m.token_tf(enc, layer, False)
    tauS = np.load(paths.ART / f"tauS_{enc}.npz", allow_pickle=True)
    li = list(tauS["layers"]).index(layer)
    H = tf(torch.as_tensor(tauS["states"][:, li], device=DEV)).half()
    row = torch.as_tensor(tauS["row"].astype(np.int64), device=DEV)
    P = profile_maxpool(H, row, int(tauS["n_rows"]), E)
    out = {}
    for t in targets:
        tau = find_tau(P, t)
        out[t] = dict(tau=tau, mean_nnz=float((P > tau).sum(1).float().mean()),
                      token_percentile=float((P.reshape(-1) < tau).float().mean() * 100))
    del P, H, E
    torch.cuda.empty_cache()
    return out


def query_taus(qval, targets=(30, 15), n_fit=5000, seed=3):
    g = rng("tau-q", seed)
    sel = g.choice(qval.shape[0], size=min(n_fit, qval.shape[0]), replace=False)
    V = torch.as_tensor(np.asarray(qval[np.sort(sel)], np.float32), device=DEV)
    out = {}
    for t in targets:
        lo, hi = -0.5, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if float((V > mid).sum(1).float().mean()) > t:
                lo = mid
            else:
                hi = mid
        tau = (lo + hi) / 2
        out[t] = dict(tau=tau, mean_nnz=float((V > tau).sum(1).float().mean()))
    return out


def our_scores_for(docs_local, qi, qv, di, dv, tau_q, tau_d, k_q, k_d, sat):
    """Our score for an explicit (query, doc) list -- used for score preservation."""
    nq = qi.shape[0]
    out = np.zeros(docs_local.shape, np.float32)
    for q in range(nq):
        w = np.maximum(qv[q, :k_q] - tau_q, 0)
        if sat == "log1p":
            w = np.log1p(w)
        qmap = {int(e): float(x) for e, x in zip(qi[q, :k_q], w) if x > 0}
        if not qmap:
            continue
        for r, d in enumerate(docs_local[q]):
            if d < 0:
                continue
            e = di[d, :k_d]; p = np.maximum(np.asarray(dv[d, :k_d], np.float32) - tau_d, 0)
            if sat == "log1p":
                p = np.log1p(p)
            s = 0.0
            for ee, pp in zip(e, p):
                if pp > 0:
                    s += qmap.get(int(ee), 0.0) * pp
            out[q, r] = s
    return out


def main(enc, layer, rep, r1_prefix, k=1000, fast=False):
    pids = np.load(paths.PREP / "c1_pids.npy")
    loc = {int(p): i for i, p in enumerate(pids)}
    qids, qtexts = load_queries(paths.QUERIES_DEV_SMALL)
    qr = {int(a): set(int(x) for x in b) for a, b in qrels_dict(paths.QRELS_DEV_SMALL).items()}
    di, dv = store(enc, layer, rep, "d")
    qi, qv = store(enc, layer, rep, "q")
    lg.info(f"C1 store: docs {di.shape}, queries {qi.shape}")

    with Timer("fit taus", lg):
        td = doc_taus(enc, layer, rep, r1_prefix)
        tq = query_taus(np.asarray(qv))
    lg.info(f"tau_d: { {k2: round(v['tau'],4) for k2,v in td.items()} } "
            f"tau_q: { {k2: round(v['tau'],4) for k2,v in tq.items()} }")

    with Timer("build inverted index", lg):
        ix = InvertedIndex(np.asarray(di), np.asarray(dv), 30000)

    res = dict(encoder=enc, layer=layer, rep=rep, tau_d=td, tau_q=tq,
               n_docs=int(len(pids)), grid={})
    # natural sparsity (§C.2.5) measured where it is uncensored: the S sample
    for t in td:
        res["tau_d"][t]["c1_nnz"] = nnz_stats(np.asarray(dv[::37], np.float32),
                                              td[t]["tau"])
    for t in tq:
        res["tau_q"][t]["nnz"] = nnz_stats(np.asarray(qv, np.float32), tq[t]["tau"])

    qi_np, qv_np = np.asarray(qi), np.asarray(qv, np.float32)
    cells = [(a, b, c, s) for a in GRID_KD for b in GRID_KQ
             for c in [(120, 30), (60, 15)] for s in GRID_SAT]
    if fast:
        cells = [c for c in cells if c[0] == 128 and c[1] == 32]
    best = None
    for k_d, k_q, (nd, nq_), sat in cells:
        name = f"kd{k_d}_kq{k_q}_nnz{nd}_{sat}"
        with Timer(f"cell {name}", lg):
            docs, scores = ix.search(qi_np, qv_np, tq[nq_]["tau"], td[nd]["tau"],
                                     k_q, k_d, sat, k=k)
        run = {int(q): pids[docs[i]] for i, q in enumerate(qids)}
        m, per = evaluate(run, qr)
        m["mrr@10_ci"] = bootstrap_ci(per["mrr@10"])
        res["grid"][name] = m
        np.savez(paths.RUNS / f"pilotC_{enc}_L{layer}_{rep}_{name}.npz",
                 qids=np.asarray(qids), docs=pids[docs].astype(np.int32), scores=scores)
        lg.info(f"{name}: MRR@10={m['mrr@10']:.4f} R@100={m['r@100']:.4f} "
                f"R@1000={m['r@1000']:.4f}")
        if best is None or m["mrr@10"] > best[1]["mrr@10"]:
            best = (name, m, (k_d, k_q, nd, nq_, sat))
    res["best"] = dict(name=best[0], **best[1])
    res["best_cfg"] = dict(k_d=best[2][0], k_q=best[2][1], nnz_d=best[2][2],
                           nnz_q=best[2][3], saturation=best[2][4])
    save_json(res, paths.RESULTS / f"31_pilotC_{enc}_L{layer}_{rep}.json")
    lg.info(f"best cell: {best[0]} MRR@10={best[1]['mrr@10']:.4f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--rep", default="R2")
    ap.add_argument("--r1-prefix", default="doc")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    main(a.encoder, a.layer, a.rep, a.r1_prefix, fast=a.fast)
