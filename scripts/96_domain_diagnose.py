"""Why does inserting a corpus's own terminology help on some (corpus, backbone) pairs and
hurt on others? Decompose the retrieval score into its trained and inserted halves.

For one (corpus, model) cell this reports, all on the same encode the evaluation used:

  A  per-entry document/query firing rate, trained vs inserted
  B  hubs: inserted entries that fire on a large share of the corpus
  C  score decomposition -- what fraction of a retrieved document's score comes from
     inserted entries, for relevant vs irrelevant documents
  D  per-query outcome vs inserted score share (does insertion hurt the queries whose
     scores it dominates?)
  E  the individual entries doing the most damage: inserted entries ranked by how much
     score mass they put on IRRELEVANT documents

Writes results/96_domain_diag_{corpus}_{model}.json.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening, precision
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("domdiag", "96_domain_diagnose.log")
DEV = "cuda"


def main(a):
    import importlib.util as iu
    spec = iu.spec_from_file_location("d92", paths.REPO / "scripts" / "92_domain_eval.py")
    d92 = iu.module_from_spec(spec); spec.loader.exec_module(d92)
    col, qids, qtexts, qrels = d92.load_domain(a.name)
    tag = f"dom_{a.name}"
    z = np.load(paths.vocab_file(tag))
    words = [str(w) for w in z["words"]]

    spec2 = iu.spec_from_file_location("e43", paths.REPO / "scripts" / "43_encode_eval.py")
    e43 = iu.module_from_spec(spec2); spec2.loader.exec_module(e43)
    cfg, head, enc, E_base, AB_base = e43.load_ckpt(a.name_model)
    layer = cfg["layer"]
    li = paths.layers_for(cfg["encoder"]).index(layer)
    blob = torch.load(paths.CKPT / a.name_model / "head.pt", map_location="cpu", weights_only=False)
    pz = np.load(paths.proto_file(cfg["encoder"], tag))
    dom = z["is_domain"] & (pz["cntA"] > 0)
    Vd = torch.as_tensor(np.asarray(pz["protoA"][:, li][dom], np.float32))
    mu, W = torch.as_tensor(blob["muV"]), torch.as_tensor(blob["WV"])
    Ed = F.normalize((Vd - mu) @ W, dim=-1).to(DEV)
    E = torch.cat([E_base, Ed], 0)
    nB, nI = E_base.shape[0], Ed.shape[0]
    ins_words = [w for w, d in zip(words, dom) if d]
    lg.info(f"{a.name}/{a.name_model}: {nB} trained + {nI} inserted entries, layer {layer}")

    mu_h, Wh = whitening.load(paths.ART / "whiten" / f"{cfg['encoder']}_L{layer}_H.npz")
    tfH = whitening.Transform("whitened", mu_h, Wh, DEV)
    mu_q, Wq = whitening.load(paths.ART / "whiten" / f"{cfg['encoder']}_L{layer}_Q.npz")
    tfQ = whitening.Transform("whitened", mu_q, Wq, DEV)
    with Timer(f"encode {a.name} (extended)", lg):
        di, dv = d92.encode(col.texts(range(len(col))), enc, head, tfH, E, None, layer, False, a.cap)
        qi, qv = d92.encode(qtexts, enc, head, tfQ, E, None, layer, True, 256)
    del enc
    torch.cuda.empty_cache()

    nV = E.shape[0]
    N, NQ = len(col), len(qids)
    rel = {r: {int(d) for d in qrels.get(q, set())} for r, q in enumerate(qids)}

    # ---- A/B: firing rates -------------------------------------------------------------
    dfreq = np.bincount(di[dv > 0].ravel(), minlength=nV).astype(np.float64) / N
    qfreq = np.bincount(qi[qv > 0].ravel(), minlength=nV).astype(np.float64) / NQ
    tr, ins = slice(0, nB), slice(nB, nV)
    res = dict(corpus=a.name, model=a.name_model, encoder=cfg["encoder"], layer=layer,
               n_trained=nB, n_inserted=nI, n_docs=N, n_queries=NQ)
    res["firing"] = {
        "trained_df_mean": float(dfreq[tr].mean()), "inserted_df_mean": float(dfreq[ins].mean()),
        "trained_df_median": float(np.median(dfreq[tr])), "inserted_df_median": float(np.median(dfreq[ins])),
        "trained_qf_mean": float(qfreq[tr].mean()), "inserted_qf_mean": float(qfreq[ins].mean()),
        "inserted_never_fires": int((dfreq[ins] == 0).sum()),
        "inserted_df_gt_50pct": int((dfreq[ins] > 0.5).sum()),
        "inserted_df_gt_20pct": int((dfreq[ins] > 0.2).sum()),
        "trained_df_gt_20pct": int((dfreq[tr] > 0.2).sum()),
        "inserted_df_gt_20pct_share_of_mass": None}

    # ---- C/D/E: score decomposition ----------------------------------------------------
    Q = torch.zeros(NQ, nV, device=DEV)
    Q.scatter_(1, torch.as_tensor(qi.astype(np.int64), device=DEV), torch.as_tensor(qv, device=DEV))
    K = min(100, N)
    best_v = torch.full((NQ, K), -1.0, device=DEV); best_i = torch.zeros((NQ, K), dtype=torch.long, device=DEV)
    best_ins = torch.zeros((NQ, K), device=DEV)
    # per-inserted-entry score mass on relevant vs irrelevant documents
    mass_rel = torch.zeros(nI, device=DEV); mass_irr = torch.zeros(nI, device=DEV)
    relmask = torch.zeros(NQ, N, dtype=torch.bool)
    for r, s in rel.items():
        if s: relmask[r, list(s)] = True
    for s0 in range(0, N, a.dchunk):
        e0 = min(s0 + a.dchunk, N)
        D = torch.zeros(e0 - s0, nV, device=DEV)
        D.scatter_(1, torch.as_tensor(di[s0:e0].astype(np.int64), device=DEV),
                   torch.as_tensor(dv[s0:e0], device=DEV))
        S_ins = Q[:, nB:] @ D[:, nB:].T
        S = Q[:, :nB] @ D[:, :nB].T + S_ins
        rm = relmask[:, s0:e0].to(DEV)
        # contribution of each inserted entry, split by relevance of the document it lands on
        contrib = Q[:, nB:].unsqueeze(1) * D[:, nB:].unsqueeze(0)      # (NQ, chunk, nI) too big
        del contrib
        for r0 in range(0, NQ, 8):
            r1 = min(r0 + 8, NQ)
            c = Q[r0:r1, nB:].unsqueeze(1) * D[:, nB:].unsqueeze(0)     # (b, chunk, nI)
            m = rm[r0:r1].unsqueeze(2)
            mass_rel += (c * m).sum((0, 1)); mass_irr += (c * ~m).sum((0, 1))
            del c
        v, i = torch.topk(S, min(K, e0 - s0), dim=1)
        ins_at = torch.gather(S_ins, 1, i)
        cv = torch.cat([best_v, v], 1); ci = torch.cat([best_i, i + s0], 1); cm = torch.cat([best_ins, ins_at], 1)
        o = torch.topk(cv, K, dim=1).indices
        best_v, best_i, best_ins = torch.gather(cv, 1, o), torch.gather(ci, 1, o), torch.gather(cm, 1, o)
        del D, S, S_ins
    ranked = best_i.cpu().numpy(); tot = best_v.cpu().numpy(); insc = best_ins.cpu().numpy()

    share_rel, share_irr, rr = [], [], []
    for r in range(NQ):
        R = rel.get(r, set())
        if not R: continue
        rr.append(next((1.0 / (k + 1) for k, d in enumerate(ranked[r][:10]) if int(d) in R), 0.0))
        for k in range(min(10, K)):
            s = insc[r, k] / max(tot[r, k], 1e-9)
            (share_rel if int(ranked[r, k]) in R else share_irr).append(float(s))
    res["score_share"] = {
        "inserted_share_top10_relevant": float(np.mean(share_rel)) if share_rel else None,
        "inserted_share_top10_irrelevant": float(np.mean(share_irr)) if share_irr else None,
        "n_rel_in_top10": len(share_rel), "n_irr_in_top10": len(share_irr)}
    mr, mi = mass_rel.cpu().numpy(), mass_irr.cpu().numpy()
    order = np.argsort(-(mi - mr))
    res["worst_entries"] = [dict(word=ins_words[j], df=float(dfreq[nB + j]), qf=float(qfreq[nB + j]),
                                 mass_irrelevant=float(mi[j]), mass_relevant=float(mr[j]))
                            for j in order[:25]]
    res["best_entries"] = [dict(word=ins_words[j], df=float(dfreq[nB + j]), qf=float(qfreq[nB + j]),
                                mass_irrelevant=float(mi[j]), mass_relevant=float(mr[j]))
                           for j in np.argsort(-(mr - mi))[:15]]
    res["mass"] = {"inserted_on_relevant": float(mr.sum()), "inserted_on_irrelevant": float(mi.sum()),
                   "ratio_irr_to_rel": float(mi.sum() / max(mr.sum(), 1e-9))}
    top = np.argsort(-dfreq[nB:])[:20]
    res["top_firing_inserted"] = [dict(word=ins_words[j], df=float(dfreq[nB + j]),
                                       qf=float(qfreq[nB + j])) for j in top]
    lg.info(f"{a.name}/{a.name_model}: inserted df mean {res['firing']['inserted_df_mean']:.4f} "
            f"vs trained {res['firing']['trained_df_mean']:.4f} | hubs>20% {res['firing']['inserted_df_gt_20pct']} "
            f"| inserted share of top-10 score: rel {res['score_share']['inserted_share_top10_relevant']} "
            f"irr {res['score_share']['inserted_share_top10_irrelevant']}")
    save_json(res, paths.RESULTS / f"96_domain_diag_{a.name}_{a.name_model}.json")
    # per-entry ground truth for every inserted entry, for the geometry analysis (98)
    np.savez(paths.RESULTS / f"96_domain_diag_{a.name}_{a.name_model}_entries.npz",
             words=np.asarray(ins_words), mass_rel=mr, mass_irr=mi,
             df=dfreq[nB:], qf=qfreq[nB:], dom_idx=np.flatnonzero(dom))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="scifact")
    ap.add_argument("--name-model", default="V1oracle")
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--dchunk", type=int, default=4000)
    main(ap.parse_args())
