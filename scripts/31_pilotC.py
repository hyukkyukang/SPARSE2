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


def store(enc, layer, rep, kind, transform="whitened"):
    sfx = "" if transform == "whitened" else f"_{transform}"
    b = paths.EMB / f"c1_{enc}_L{layer}_{rep}{sfx}"
    return (np.load(b.with_name(b.name + f"_{kind}_idx.npy"), mmap_mode="r"),
            np.load(b.with_name(b.name + f"_{kind}_val.npy"), mmap_mode="r"))


def doc_taus(enc, layer, rep, r1_prefix, targets=(120, 60), transform="whitened"):
    """tau_d fitted on the 5,000 S passages of §0.6, in this exact space."""
    sys.path.insert(0, str(paths.REPO / "scripts"))
    import importlib.util as iu
    spec = iu.spec_from_file_location("e30", paths.REPO / "scripts" / "30_encode_c1.py")
    m = iu.module_from_spec(spec); spec.loader.exec_module(m)
    E = m.entry_matrix(enc, layer, rep, r1_prefix, transform)
    tf = m.token_tf(enc, layer, False, transform)
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


def our_scores_for(docs_local, qi, qv, di, dv, tau_q, tau_d, k_q, k_d, sat, n_entries=30000):
    """Our score for an explicit (query, doc) list -- used for score preservation.

    Scatters the query into a dense |V| vector once, then gathers each candidate
    document's own entries out of it: 100 documents per query cost one gather.
    """
    nq = qi.shape[0]
    out = np.zeros(docs_local.shape, np.float32)
    qbuf = np.zeros(n_entries, np.float32)
    for q in range(nq):
        w = np.maximum(np.asarray(qv[q, :k_q], np.float32) - tau_q, 0)
        if sat == "log1p":
            w = np.log1p(w)
        ids = np.asarray(qi[q, :k_q])
        qbuf[:] = 0.0
        np.add.at(qbuf, ids[w > 0], w[w > 0])
        sel = docs_local[q]
        ok = sel >= 0
        if not ok.any():
            continue
        d_ids = np.asarray(di[sel[ok]][:, :k_d])
        d_w = np.maximum(np.asarray(dv[sel[ok]][:, :k_d], np.float32) - tau_d, 0)
        if sat == "log1p":
            d_w = np.log1p(d_w)
        out[q, ok] = (qbuf[d_ids] * d_w).sum(1)
    return out


def main(enc, layer, rep, r1_prefix, k=1000, fast=False, transform="whitened"):
    pids = np.load(paths.PREP / "c1_pids.npy")
    loc = {int(p): i for i, p in enumerate(pids)}
    qids, qtexts = load_queries(paths.QUERIES_DEV_SMALL)
    qr = {int(a): set(int(x) for x in b) for a, b in qrels_dict(paths.QRELS_DEV_SMALL).items()}
    di, dv = store(enc, layer, rep, "d", transform)
    qi, qv = store(enc, layer, rep, "q", transform)
    lg.info(f"C1 store: docs {di.shape}, queries {qi.shape}")

    with Timer("fit taus", lg):
        td = doc_taus(enc, layer, rep, r1_prefix, transform=transform)
        tq = query_taus(np.asarray(qv))
    lg.info(f"tau_d: { {k2: round(v['tau'],4) for k2,v in td.items()} } "
            f"tau_q: { {k2: round(v['tau'],4) for k2,v in tq.items()} }")

    with Timer("build inverted index", lg):
        ix = InvertedIndex(np.asarray(di), np.asarray(dv), 30000)

    res = dict(encoder=enc, layer=layer, rep=rep, transform=transform,
               tau_d=td, tau_q=tq,
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
        np.savez(paths.RUNS / f"pilotC_{enc}_L{layer}_{rep}_{transform}_{name}.npz",
                 qids=np.asarray(qids), docs=pids[docs].astype(np.int32), scores=scores)
        lg.info(f"{name}: MRR@10={m['mrr@10']:.4f} R@100={m['r@100']:.4f} "
                f"R@1000={m['r@1000']:.4f}")
        if best is None or m["mrr@10"] > best[1]["mrr@10"]:
            best = (name, m, (k_d, k_q, nd, nq_, sat))
    # ---- score preservation vs the dense encoder (§C.2.4) -------------------
    k_d, k_q, nd, nq_, sat = best[2]
    dpath = paths.RUNS / f"dense_{enc}_c1.npz"
    if dpath.exists():
        dz = np.load(dpath)
        d_qids, d_docs, d_scores = dz["qids"], dz["docs"], dz["scores"]
        dq = {int(q): i for i, q in enumerate(d_qids)}
        sp_rho, jac = [], []
        ours_run = np.load(paths.RUNS /
                           f"pilotC_{enc}_L{layer}_{rep}_{transform}_{best[0]}.npz")["docs"]
        for i, q in enumerate(qids):
            r = dq.get(int(q))
            if r is None:
                continue
            top = d_docs[r][:100]
            dsc = d_scores[r][:100]
            local = np.array([loc.get(int(p), -1) for p in top])
            osc = our_scores_for(local[None, :], qi_np[i:i + 1], qv_np[i:i + 1],
                                 di, dv, tq[nq_]["tau"], td[nd]["tau"], k_q, k_d, sat)[0]
            m = local >= 0
            if m.sum() > 5:
                sp_rho.append(sps.spearmanr(dsc[m], osc[m]).statistic)
            jac.append(len(set(top.tolist()) & set(ours_run[i][:100].tolist())) /
                       len(set(top.tolist()) | set(ours_run[i][:100].tolist())))
        res["score_preservation"] = dict(
            spearman_mean=float(np.nanmean(sp_rho)),
            spearman_median=float(np.nanmedian(sp_rho)),
            jaccard_top100=float(np.mean(jac)), n=int(len(sp_rho)))
        lg.info(f"score preservation: spearman={np.nanmean(sp_rho):.3f} "
                f"jaccard@100={np.mean(jac):.3f}")

    # ---- qualitative dump (§C.2.6) and SPLADE overlap (§C.2.7) ---------------
    z = np.load(paths.ART / "vocab.npz")
    words = [str(w) for w in z["words"]]
    col = Collection()
    g = rng("qualitative")
    dsel = np.sort(g.choice(len(pids), size=20, replace=False))
    qsel = np.sort(g.choice(len(qids), size=20, replace=False))
    tag2 = "" if transform == "whitened" else f"_{transform}"
    lines = ["# Pilot C — qualitative top-10 entries", "",
             f"encoder={enc} layer={layer} rep={rep} cell={best[0]}", ""]
    for tag, sel, I, V, texts, tau in (
            ("passage", dsel, di, dv, [col[int(pids[i])] for i in dsel], td[nd]["tau"]),
            ("query", qsel, qi, qv, [qtexts[i] for i in qsel], tq[nq_]["tau"])):
        lines.append(f"## {tag}s")
        for r, i in enumerate(sel):
            w = np.maximum(np.asarray(V[i][:10], np.float32) - tau, 0)
            terms = ", ".join(f"{words[int(e)]}:{x:.2f}"
                              for e, x in zip(I[i][:10], w) if x > 0)
            lines.append(f"- *{texts[r][:150]}*")
            lines.append(f"  - {terms}")
        lines.append("")
    (paths.REPORTS / f"pilotC_qualitative_{enc}_L{layer}_{rep}{tag2}.md").write_text(
        "\n".join(lines))

    exp = np.load(paths.ART / "splade_expansions.npz", allow_pickle=True)
    sp_vocab = [str(x) for x in exp["vocab"]]
    inv = {w: i for i, w in enumerate(words)}
    sp_set = set(sp_vocab)
    in_splade = np.array([w in sp_set for w in words])
    pid2e = {int(p): i for i, p in enumerate(exp["pids"])}
    jj, cc = [], []
    ov = [int(p) for p in exp["overlap_pids"] if int(p) in loc]
    for p in ov[:1000]:
        d = loc[p]
        ids = np.asarray(di[d][:k_d])
        w = np.maximum(np.asarray(dv[d][:k_d], np.float32) - td[nd]["tau"], 0)
        order = np.argsort(-w)[:20]
        ours = {int(ids[o]) for o in order if w[o] > 0 and in_splade[ids[o]]}
        toks = [sp_vocab[t] for t in exp["idx"][pid2e[p]][:20]]
        S = {inv[t] for t in toks if t in inv}
        cc.append(len(S) / 20)
        if S or ours:
            jj.append(len(S & ours) / max(len(S | ours), 1))
    res["splade_overlap_truncated"] = dict(jaccard=float(np.mean(jj)),
                                           coverage=float(np.mean(cc)), n=len(jj))
    lg.info(f"SPLADE overlap (truncated): J={np.mean(jj):.3f} cov={np.mean(cc):.2f}")

    res["best"] = dict(name=best[0], **best[1])
    res["best_cfg"] = dict(k_d=best[2][0], k_q=best[2][1], nnz_d=best[2][2],
                           nnz_q=best[2][3], saturation=best[2][4])
    tag = "" if transform == "whitened" else f"_{transform}"
    save_json(res, paths.RESULTS / f"31_pilotC_{enc}_L{layer}_{rep}{tag}.json")
    lg.info(f"best cell: {best[0]} MRR@10={best[1]['mrr@10']:.4f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--rep", default="R2")
    ap.add_argument("--r1-prefix", default="doc")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--transform", default="whitened",
                    choices=["raw", "centered", "whitened"])
    a = ap.parse_args()
    main(a.encoder, a.layer, a.rep, a.r1_prefix, fast=a.fast, transform=a.transform)
