"""Pilot D evaluation (§D.6): recovery ratio, Q_H, activation statistics, C-alias."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import load_queries, load_qrels, qrels_dict, Collection, load_splits
from dvlsr.metrics import evaluate, bootstrap_ci, ratio_bootstrap, paired_bootstrap
from dvlsr.retrieval import InvertedIndex
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("evalD", "44_pilotD_eval.log")
DEV = "cuda"


def store(name, kind):
    b = paths.EMB / f"D_{name}_{kind}"
    return (np.load(b.with_name(b.name + "_idx.npy"), mmap_mode="r"),
            np.load(b.with_name(b.name + "_val.npy"), mmap_mode="r"))


def masked_queries(qi, qv, held_mask, mode="zero", alias=None):
    """'zero' removes held-out entries from the query side (== zeroing both sides).
    'alias' remaps each held-out entry to its nearest seen entry (control C-alias)."""
    qi2, qv2 = np.array(qi), np.array(qv, np.float32)
    h = held_mask[qi2]
    if mode == "zero":
        qv2[h] = 0.0
    elif mode == "alias":
        qi2[h] = alias[qi2[h]]
    return qi2, qv2


def run_search(ix, qi, qv, k=1000, k_d=1024, k_q=256):
    return ix.search(qi, qv, tau_q=0.0, tau_d=0.0, k_q=k_q, k_d=k_d,
                     saturation="none", k=k)


def qh_primary(oracle_q, oracle_d, held_mask, qids, qrels, loc, thresh=0.2):
    """Dev queries where held-out entries carry >= 20% of the ORACLE query-positive score."""
    qi, qv = oracle_q
    di, dv = oracle_d
    keep, ratios = [], []
    for r, q in enumerate(qids):
        pos = [p for p in qrels.get(int(q), []) if int(p) in loc]
        if not pos:
            continue
        qmap = {}
        for e, v in zip(qi[r], np.asarray(qv[r], np.float32)):
            if v > 0:
                qmap[int(e)] = float(v)
        tot = held = 0.0
        for p in pos:
            d = loc[int(p)]
            for e, v in zip(di[d], np.asarray(dv[d], np.float32)):
                w = qmap.get(int(e))
                if w and v > 0:
                    tot += w * float(v)
                    if held_mask[int(e)]:
                        held += w * float(v)
        rr = held / tot if tot > 0 else 0.0
        ratios.append(rr)
        if rr >= thresh:
            keep.append(int(q))
    return np.asarray(keep), np.asarray(ratios)


def qh_lex(qids, qtexts, qrels, held_mask, words, loc):
    """Secondary, easier definition: a held-out word occurs in query and a relevant passage."""
    import re
    col = Collection()
    WR = re.compile(r"[A-Za-z0-9]+")
    inv = {w: i for i, w in enumerate(words)}
    out = []
    for r, q in enumerate(qids):
        qt = {t.lower() for t in WR.findall(qtexts[r]) if t.isalpha()}
        hq = {t for t in qt if (j := inv.get(t)) is not None and held_mask[j]}
        if not hq:
            continue
        for p in qrels.get(int(q), []):
            dt = {t.lower() for t in WR.findall(col[int(p)]) if t.isalpha()}
            if hq & dt:
                out.append(int(q)); break
    return np.asarray(out)


def activation_stats(name, held_mask, decile, n_docs=None):
    """§D.6.5 activation statistics on S, compared held vs seen *within decile*."""
    si, sv = store(name, "s")
    n = si.shape[0] if n_docs is None else min(n_docs, si.shape[0])
    nV = len(held_mask)
    df = np.zeros(nV, np.int64)
    wsum = np.zeros(nV, np.float64)
    B = 20000
    for s in range(0, n, B):
        i = np.asarray(si[s:s + B]); v = np.asarray(sv[s:s + B], np.float32)
        m = v > 0
        df += np.bincount(i[m], minlength=nV)
        wsum += np.bincount(i[m], weights=v[m], minlength=nV)
    r = df / n
    wbar = wsum / np.maximum(df, 1)
    out = dict(n_docs=int(n), mean_nnz=float(df.sum() / n))
    per_dec, wass = {}, []
    logs, logw = [], []
    for d in range(10):
        sel = decile == d
        hs, ss = sel & held_mask, sel & ~held_mask
        if hs.sum() == 0 or ss.sum() == 0:
            continue
        mh, ms = float(np.median(r[hs])), float(np.median(r[ss]))
        wh, ws = float(np.median(wbar[hs])), float(np.median(wbar[ss]))
        lr = float(np.log((mh + 1e-9) / (ms + 1e-9)))
        lw = float(np.log((wh + 1e-9) / (ws + 1e-9)))
        logs.append(lr); logw.append(lw)
        w = float(sps.wasserstein_distance(r[hs], r[ss]))
        wass.append(w)
        per_dec[int(d)] = dict(r_held=mh, r_seen=ms, log_ratio=lr,
                               w_held=wh, w_seen=ws, log_w_ratio=lw, wasserstein=w,
                               n_held=int(hs.sum()), n_seen=int(ss.sum()))
    out.update(by_decile=per_dec,
               signed_gap_r=float(np.median(logs)) if logs else float("nan"),
               abs_gap_r=float(np.median(np.abs(logs))) if logs else float("nan"),
               gap_w=float(np.median(logw)) if logw else float("nan"),
               wasserstein=float(np.mean(wass)) if wass else float("nan"),
               dead_held=float((r[held_mask] == 0).mean()),
               dead_seen=float((r[~held_mask] == 0).mean()))
    np.save(paths.ART / f"D_actstats_{name}.npy", np.stack([r, wbar]))
    return out


def nearest_seen(name, held_mask):
    blob = torch.load(paths.CKPT / name / "head.pt", map_location=DEV, weights_only=False)
    E = blob["E_all"].to(DEV).float()
    seen = torch.as_tensor(np.flatnonzero(~held_mask), device=DEV)
    alias = np.arange(len(held_mask))
    H = torch.as_tensor(np.flatnonzero(held_mask), device=DEV)
    for s in range(0, len(H), 2048):
        idx = H[s:s + 2048]
        sim = E[idx] @ E[seen].T
        alias[idx.cpu().numpy()] = seen[sim.argmax(1)].cpu().numpy()
    return alias


def evaluate_model(name, oracle_name, split, k=1000, do_alias=True):
    z = np.load(paths.ART / "vocab.npz")
    words = [str(w) for w in z["words"]]
    decile = z["decile"]
    sp = np.load(paths.PREP / "vocab_splits.npz")
    held = sp[split]
    pids = np.load(paths.PREP / "c1_pids.npy")
    loc = {int(p): i for i, p in enumerate(pids)}
    qids, qtexts = load_queries(paths.QUERIES_DEV_SMALL)
    qrels = {int(a): set(int(x) for x in b)
             for a, b in qrels_dict(paths.QRELS_DEV_SMALL).items()}

    res = dict(name=name, oracle=oracle_name, split=split, n_held=int(held.sum()))
    di, dv = store(name, "d")
    qi, qv = store(name, "q")
    res["cap_hit_frac"] = float((np.asarray(dv[::53]) > 0).sum(1).mean() / dv.shape[1] >= 1.0)
    res["nnz_d"] = float((np.asarray(dv[::53], np.float32) > 0).sum(1).mean())
    res["nnz_q"] = float((np.asarray(qv, np.float32) > 0).sum(1).mean())

    with Timer(f"index {name}", lg):
        ix = InvertedIndex(np.asarray(di), np.asarray(dv), len(words), min_val=0.0)
    per_q = {}
    runs = {}
    variants = [("all", "keep"), ("seen", "zero")] + ([("alias", "alias")] if do_alias else [])
    alias = nearest_seen(name, held) if do_alias else None
    for tag, mode in variants:
        q2i, q2v = (np.asarray(qi), np.asarray(qv, np.float32)) if mode == "keep" \
            else masked_queries(qi, qv, held, mode, alias)
        docs, _ = run_search(ix, q2i, q2v, k=k)
        run = {int(q): pids[docs[i]] for i, q in enumerate(qids)}
        m, per = evaluate(run, qrels)
        m["mrr@10_ci"] = bootstrap_ci(per["mrr@10"])
        res[f"overall_{tag}"] = m
        per_q[tag] = per
        runs[tag] = docs
        lg.info(f"{name} [{tag}]: MRR@10={m['mrr@10']:.4f} R@100={m['r@100']:.4f}")
    del ix
    torch.cuda.empty_cache()
    return res, per_q, held, decile, qids, qtexts, qrels, loc, words


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--no-alias", action="store_true")
    a = ap.parse_args()
    res, per_q, held, decile, qids, qtexts, qrels, loc, words = evaluate_model(
        a.name, a.oracle, a.split, do_alias=not a.no_alias)
    # oracle side
    ores, o_per, *_ = evaluate_model(a.oracle, a.oracle, a.split, do_alias=False)
    QH, ratios = qh_primary(store(a.oracle, "q"), store(a.oracle, "d"), held,
                            qids, qrels, loc)
    QHl = qh_lex(qids, qtexts, qrels, held, words, loc)
    res["Q_H"] = dict(n=int(len(QH)), n_lex=int(len(QHl)),
                      mean_held_ratio=float(np.mean(ratios)))
    lg.info(f"|Q_H| = {len(QH)}  |Q_H-lex| = {len(QHl)}")
    qpos = {int(q): i for i, q in enumerate(per_q["all"]["qids"])}
    for label, Qset in (("Q_H", QH), ("Q_H_lex", QHl)):
        sel = np.array([qpos[q] for q in Qset if q in qpos], dtype=int)
        if len(sel) == 0:
            continue
        ma, ms = per_q["all"]["mrr@10"][sel], per_q["seen"]["mrr@10"][sel]
        oa, os_ = o_per["all"]["mrr@10"][sel], o_per["seen"]["mrr@10"][sel]
        rho = ratio_bootstrap(ma, ms, oa, os_)
        den = paired_bootstrap(oa, os_)
        entry = dict(n=int(len(sel)), model_all=float(ma.mean()), model_seen=float(ms.mean()),
                     oracle_all=float(oa.mean()), oracle_seen=float(os_.mean()),
                     rho=rho, denominator=den,
                     denominator_valid=bool(den["diff"] >= 0.02 and den["excludes_zero"]),
                     gap_to_oracle_rel=float((oa.mean() - ma.mean()) / max(oa.mean(), 1e-9)))
        if "alias" in per_q:
            al = per_q["alias"]["mrr@10"][sel]
            entry["alias_mrr"] = float(al.mean())
            entry["rho_alias"] = ratio_bootstrap(al, ms, oa, os_)
            entry["vs_alias"] = paired_bootstrap(ma, al)
        res[label] = entry
        lg.info(f"{label}: rho={rho['rho']:.3f} [{rho['lo']:.2f},{rho['hi']:.2f}] "
                f"den={den['diff']:.4f} valid={entry['denominator_valid']}")
    with Timer("activation statistics", lg):
        res["activation"] = activation_stats(a.name, held, decile)
        res["activation_oracle"] = activation_stats(a.oracle, held, decile)
    lg.info(f"signed gap_r = {res['activation']['signed_gap_r']:.4f} "
            f"|gap_r| = {res['activation']['abs_gap_r']:.4f}")
    save_json(res, paths.RESULTS / f"44_pilotD_{a.name}.json")
