"""What actually distinguishes an over-firing held-out entry? (Pilot F diagnostic)

Pilot F normalises each entry's background mean and standard deviation. That is only
the right correction if those moments are what differ between seen and held-out
entries. This script measures, under a *trained* model:

  mu_j, sd_j     mean and sd of cos(g(h~), v~_j) over the document-side token bank
  q_j            a high quantile of the same distribution (what max-pooling actually
                 selects: s_j is driven by the upper tail, not the centre)
  r_j            the realised activation rate over the statistics sample S, taken from
                 the file 44_pilotD_eval already wrote (max-pooled and thresholded)

and reports the seen/held gap of each **within frequency decile**, plus how well each
statistic explains r_j. If the rate gap is large while the mu/sd gap is ~0, then a
z-normalisation of the moments cannot fix the rate and the correction has to target
the tail instead.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("entrystats", "48_entry_stats.py.log")
DEV = "cuda"


def stats_for(name, n_bank=65536, chunk_rows=2048, chunk_ent=10000, q=0.999):
    import importlib.util as iu
    spec = iu.spec_from_file_location("e43", paths.REPO / "scripts" / "43_encode_eval.py")
    e43 = iu.module_from_spec(spec); spec.loader.exec_module(e43)
    cfg, head, enc, E, AB = e43.load_ckpt(name)
    del enc
    torch.cuda.empty_cache()
    layer = cfg["layer"]
    mu_h, W = whitening.load(paths.ART / "whiten" / f"{cfg['encoder']}_L{layer}_H.npz")
    tf = whitening.Transform("whitened", mu_h, W, DEV)
    bank = np.load(paths.ART / f"bankH_{cfg['encoder']}.npy", mmap_mode="r")
    li = paths.layers_for(cfg["encoder"]).index(layer)
    g = np.random.default_rng(0)
    sel = np.sort(g.choice(bank.shape[0], size=min(n_bank, bank.shape[0]), replace=False))
    B = torch.as_tensor(np.asarray(bank[sel, li], np.float32), device=DEV)
    nV = E.shape[0]
    s1 = torch.zeros(nV, device=DEV, dtype=torch.float64)
    s2 = torch.zeros(nV, device=DEV, dtype=torch.float64)
    # exact high quantile needs the column; keep only the top-m per entry as we stream
    m = max(int((1 - q) * B.shape[0]), 8)
    top = torch.full((nV, m), -2.0, device=DEV)
    n = 0
    with torch.no_grad():
        for r in range(0, B.shape[0], chunk_rows):
            h = tf(B[r:r + chunk_rows], normalize=False)
            gh = F.normalize(head(h), dim=-1)
            n += gh.shape[0]
            for e in range(0, nV, chunk_ent):
                a = (gh @ E[e:e + chunk_ent].T).float()          # (rows, entries)
                s1[e:e + chunk_ent] += a.sum(0).double()
                s2[e:e + chunk_ent] += (a * a).sum(0).double()
                cat = torch.cat([top[e:e + chunk_ent], a.T], 1)
                top[e:e + chunk_ent] = torch.topk(cat, m, dim=1).values
                del a, cat
    mu = (s1 / n).float().cpu().numpy()
    sd = (s2 / n - (s1 / n) ** 2).clamp(min=1e-12).sqrt().float().cpu().numpy()
    qv = top[:, -1].cpu().numpy()                    # the m-th largest == the q quantile
    mx = top[:, 0].cpu().numpy()
    return cfg, mu, sd, qv, mx


def main(a):
    cfg, mu, sd, qv, mx = stats_for(a.name, n_bank=a.n_bank)
    z = np.load(paths.vocab_file(cfg.get("vocab_tag", "") or ""))
    decile = z["decile"]
    held = np.load(paths.vsplits_file(cfg.get("vocab_tag", "") or ""))[cfg["split"]]
    seen = ~held
    act = paths.ART / f"D_actstats_{a.name}.npy"
    r = np.load(act)[0] if act.exists() else None

    out = dict(name=a.name, split=cfg["split"], n_bank=a.n_bank)
    rows = []
    series = [("mu", mu), ("sd", sd), (f"q{a.q}", qv), ("max", mx)]
    if r is not None:
        series.append(("rate", r))
    for tag, v in series:
        per = []
        for d in range(10):
            m = decile == d
            h, s = m & held, m & seen
            if h.sum() == 0 or s.sum() == 0:
                continue
            mh, ms = float(np.median(v[h])), float(np.median(v[s]))
            per.append(dict(decile=d, held=mh, seen=ms, diff=mh - ms,
                            log_ratio=float(np.log((abs(mh) + 1e-9) / (abs(ms) + 1e-9)))))
        med_diff = float(np.median([p["diff"] for p in per]))
        med_log = float(np.median([p["log_ratio"] for p in per]))
        rows.append(dict(stat=tag, median_diff=med_diff, median_log_ratio=med_log,
                         held_median=float(np.median(v[held])),
                         seen_median=float(np.median(v[seen])), by_decile=per))
        lg.info(f"{tag:6s} within-decile median gap held-seen = {med_diff:+.5f} "
                f"(log ratio {med_log:+.4f})")
    out["gaps"] = rows

    if r is not None:
        # which statistic explains the realised activation rate?
        ok = r > 0
        corr = {}
        for tag, v in (("mu", mu), ("sd", sd), ("q", qv), ("max", mx)):
            corr[tag] = dict(
                spearman_all=float(sps.spearmanr(v[ok], r[ok]).statistic),
                spearman_within_decile=float(np.median([
                    sps.spearmanr(v[ok & (decile == d)], r[ok & (decile == d)]).statistic
                    for d in range(10) if (ok & (decile == d)).sum() > 50])))
        out["explains_rate"] = corr
        for k, v in corr.items():
            lg.info(f"rate ~ {k:4s}: spearman all {v['spearman_all']:+.3f}  "
                    f"within decile {v['spearman_within_decile']:+.3f}")
    save_json(out, paths.RESULTS / f"48_entry_stats_{a.name}.json")
    lg.info(f"wrote 48_entry_stats_{a.name}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-bank", type=int, default=65536)
    ap.add_argument("--q", type=float, default=0.999)
    main(ap.parse_args())
