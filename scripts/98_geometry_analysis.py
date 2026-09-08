"""Can the geometry of the inserted prototypes, in the backbone's own space, tell helpful
entries from harmful ones -- without encoding the corpus or seeing any label?

Ground truth per inserted entry comes from scripts/96_domain_diagnose.py: the retrieval-score
mass it put on relevant vs irrelevant documents. An entry is HARMFUL if it put more mass on
irrelevant documents than relevant ones (and fired at all), HELPFUL otherwise.

Candidate statistics, all computable from the new prototypes plus the frozen artifacts:
  norm_raw    L2 norm of the raw prototype (mean of contextual states): a word used in
              maximally diverse contexts averages to a short vector near the centroid
  cos_cent    cosine to the trained vocabulary's centroid, raw space and whitened space
  cnt         number of occurrences the prototype was built from
  nn_trained  cosine to the nearest trained entry (whitened, normalised)
  qH / qQ     99.9th percentile of the entry's cosine, in the model's scoring space, against
              the DOCUMENT token bank and the QUERY token bank (Pilot F's statistic, both sides)
  muH / muQ   mean of the same distributions
Aggregate, per cell: mean pairwise cosine of the inserted set and the variance share of its
top eigen-direction, against the same for an equal-sized random subset of trained entries.

Reports rank correlation with harm and AUC (harmful vs helpful) per statistic per cell.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening, precision
from dvlsr.util import get_logger, save_json

lg = get_logger("geom", "98_geometry_analysis.log")
DEV = "cuda"


def auc(score, label):
    """AUC of `score` for predicting label==1 (higher score -> more likely 1)."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    if y.all() or (~y).all(): return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # tie-aware ranks
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def bank_stats(head, E, bank_path, tf, li, n=32768, q=0.999):
    bank = np.load(bank_path, mmap_mode="r")
    g = np.random.default_rng(1)
    sel = np.sort(g.choice(bank.shape[0], size=min(n, bank.shape[0]), replace=False))
    B = torch.as_tensor(np.asarray(bank[sel, li], np.float32), device=DEV)
    mtop = max(int((1 - q) * B.shape[0]), 8)
    nV = E.shape[0]
    s1 = torch.zeros(nV, device=DEV, dtype=torch.float64)
    top = torch.full((nV, mtop), -2.0, device=DEV)
    with torch.no_grad(), precision.autocast():
        for r in range(0, B.shape[0], 4096):
            gh = F.normalize(head(tf(B[r:r + 4096], normalize=False)).float(), dim=-1)
            a = (gh @ E.T).float()
            s1 += a.sum(0).double()
            top = torch.topk(torch.cat([top, a.T], 1), mtop, dim=1).values
    return (s1 / B.shape[0]).float().cpu().numpy(), top[:, -1].cpu().numpy()


def main(a):
    import importlib.util as iu
    spec = iu.spec_from_file_location("e43", paths.REPO / "scripts" / "43_encode_eval.py")
    e43 = iu.module_from_spec(spec); spec.loader.exec_module(e43)
    cfg, head, enc, E_base, AB = e43.load_ckpt(a.name_model)
    del enc; torch.cuda.empty_cache()
    layer = cfg["layer"]; li = paths.layers_for(cfg["encoder"]).index(layer)
    tag = f"dom_{a.name}"
    z = np.load(paths.vocab_file(tag))
    pz = np.load(paths.proto_file(cfg["encoder"], tag))
    dom = z["is_domain"] & (pz["cntA"] > 0)
    gt = np.load(paths.RESULTS / f"96_domain_diag_{a.name}_{a.name_model}_entries.npz")
    assert (gt["dom_idx"] == np.flatnonzero(dom)).all(), "ground truth / vocabulary mismatch"
    words = [str(w) for w in gt["words"]]

    blob = torch.load(paths.CKPT / a.name_model / "head.pt", map_location="cpu", weights_only=False)
    muV, WV = torch.as_tensor(blob["muV"]), torch.as_tensor(blob["WV"])
    Vraw = torch.as_tensor(np.asarray(pz["protoA"][:, li][dom], np.float32))
    Ed = F.normalize((Vraw - muV) @ WV, dim=-1).to(DEV)
    Ptr = torch.as_tensor(np.asarray(np.load(paths.proto_file(cfg["encoder"]))["protoA"][:, li], np.float32))
    cnt = pz["cntA"][dom]

    st = {}
    st["norm_raw"] = Vraw.norm(dim=-1).numpy()
    cen_raw = F.normalize(Ptr.mean(0), dim=-1)
    st["cos_cent_raw"] = (F.normalize(Vraw, dim=-1) @ cen_raw).numpy()
    cen_w = F.normalize(E_base.mean(0), dim=-1)
    st["cos_cent_white"] = (Ed @ cen_w).cpu().numpy()
    st["cnt"] = cnt.astype(np.float64)
    with torch.no_grad():
        nn_ = torch.cat([(Ed[i:i + 512] @ E_base.T).max(1).values for i in range(0, Ed.shape[0], 512)])
    st["nn_trained"] = nn_.cpu().numpy()
    WD = paths.ART / "whiten"
    muH, WH = whitening.load(WD / f"{cfg['encoder']}_L{layer}_H.npz"); tfH = whitening.Transform("whitened", muH, WH, DEV)
    muQ, WQ = whitening.load(WD / f"{cfg['encoder']}_L{layer}_Q.npz"); tfQ = whitening.Transform("whitened", muQ, WQ, DEV)
    st["muH"], st["qH"] = bank_stats(head, Ed, paths.ART / f"bankH_{cfg['encoder']}.npy", tfH, li)
    st["muQ"], st["qQ"] = bank_stats(head, Ed, paths.ART / f"bankQ_{cfg['encoder']}.npy", tfQ, li)
    # the same bank statistics for the TRAINED entries give each backbone its own reference
    muH_tr, qH_tr = bank_stats(head, E_base, paths.ART / f"bankH_{cfg['encoder']}.npy", tfH, li)
    muQ_tr, qQ_tr = bank_stats(head, E_base, paths.ART / f"bankQ_{cfg['encoder']}.npy", tfQ, li)
    st["qH_rel"] = st["qH"] / np.percentile(qH_tr, 99)
    st["qQ_rel"] = st["qQ"] / np.percentile(qQ_tr, 99)

    # Ground truth: an entry is harmful if it concentrates its score mass on relevant
    # documents LESS than chance would. Absolute mass is dominated by irrelevant documents on
    # every corpus (scifact: ~1 relevant per query among 5,183), so the comparison must be
    # against the base rate of relevant (query, document) pairs, i.e. a lift.
    mr, mi = gt["mass_rel"].astype(np.float64), gt["mass_irr"].astype(np.float64)
    meta = json.load(open(paths.RESULTS / f"90_domain_{a.name}.json"))
    base = meta["n_qrels"] / (meta["n_queries"] * meta["n_passages"])
    fired = (mr + mi) > 0
    lift = np.where(fired, (mr / np.maximum(mr + mi, 1e-12)) / base, np.nan)
    harm = -np.log(np.maximum(lift, 1e-6))          # higher = more harmful
    harmful = fired & (lift < 1.0)
    helpful = fired & (lift >= 1.0)
    lab = harmful[fired]
    out = dict(corpus=a.name, model=a.name_model, encoder=cfg["encoder"], n_inserted=int(dom.sum()),
               n_fired=int(fired.sum()), n_harmful=int(harmful.sum()), n_helpful=int(helpful.sum()),
               per_stat={})
    lg.info(f"{a.name}/{cfg['encoder']}: {int(dom.sum())} inserted, {int(fired.sum())} fired, "
            f"{int(harmful.sum())} harmful, {int(helpful.sum())} helpful")
    for k, v in st.items():
        v = np.asarray(v, float)
        out["per_stat"][k] = dict(auc_harmful=auc(v[fired], lab), spearman_harm=spearman(v[fired], harm[fired]),
                                  spearman_df=spearman(v, gt["df"]), spearman_qf=spearman(v, gt["qf"]))
    # firing statistics themselves, for reference (they need the corpus / queries)
    for k in ("df", "qf"):
        v = gt[k].astype(float)
        out["per_stat"]["FIRING_" + k] = dict(auc_harmful=auc(v[fired], lab), spearman_harm=spearman(v[fired], harm[fired]))
    # aggregate geometry of the inserted set vs an equal-sized random trained subset
    def agg(X):
        X = X.float()
        G = X @ X.T; n = X.shape[0]
        mpc = float((G.sum() - G.diagonal().sum()) / (n * (n - 1)))
        Xc = X - X.mean(0)
        ev = torch.linalg.eigvalsh((Xc.T @ Xc) / (n - 1))
        return dict(mean_pairwise_cos=mpc, top_eig_share=float(ev[-1] / ev.sum()), top5_eig_share=float(ev[-5:].sum() / ev.sum()))
    g = torch.Generator().manual_seed(0)
    sub = E_base[torch.randperm(E_base.shape[0], generator=g)[:Ed.shape[0]].to(E_base.device)]
    out["aggregate"] = dict(inserted=agg(Ed), trained_subset=agg(sub))
    # the specific contrast the study needs: e5's helpful-but-broad entries vs its harmful ones
    lift_f = np.where(fired, lift, np.nan)
    idx_h = np.argsort(-np.nan_to_num(lift_f, nan=-1))[:15]; idx_b = np.argsort(np.nan_to_num(lift_f, nan=1e9))[:15]
    ex = lambda i: dict(word=words[i], lift=float(lift_f[i]), **{k: float(np.asarray(st[k])[i]) for k in ("norm_raw","cos_cent_white","qH_rel","qQ_rel","cnt")}, df=float(gt["df"][i]), qf=float(gt["qf"][i]))
    out["examples"] = {"helpful": [ex(i) for i in idx_h], "harmful": [ex(i) for i in idx_b]}
    out["base_rate"] = base
    save_json(out, paths.RESULTS / f"98_geometry_{a.name}_{a.name_model}.json")
    best = max(out["per_stat"].items(), key=lambda kv: abs(kv[1]["auc_harmful"] - 0.5) if kv[1]["auc_harmful"] == kv[1]["auc_harmful"] else 0)
    lg.info(f"{a.name}/{cfg['encoder']}: best separator {best[0]} AUC {best[1]['auc_harmful']:.3f} | "
            f"aggregate mean pairwise cos inserted {out['aggregate']['inserted']['mean_pairwise_cos']:.3f} vs trained {out['aggregate']['trained_subset']['mean_pairwise_cos']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="scifact")
    ap.add_argument("--name-model", default="V1oracle")
    main(ap.parse_args())
