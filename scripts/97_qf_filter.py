"""Drop inserted entries that fire on too many QUERIES, and re-evaluate.

`scripts/96_domain_diagnose.py` found one statistic that separates every (corpus, backbone)
cell we have: the mean query-side firing rate of the inserted entries. The four cells where
insertion helped sit at 0.0006-0.0048; the two where it hurt sit at 0.0158 and 0.0704, with
no overlap. Document-side firing does not separate them (e5 gains on scifact with the
highest doc firing of any cell and loses on trec-covid with a lower one).

The mechanism: an entry helps when it fires on the few queries that need it and on the
documents that answer them, and hurts when it fires on MANY queries, because it then adds
score to every document containing it regardless of relevance. On trec-covid, e5 fires
discourse words (`repercussions`, `evidences`, `pathophysiologic`) on ~80% of documents AND
40-88% of queries, taking inserted entries to 96% of the total score.

This filter is computable at insertion time from a sample of queries and no labels:
encode the queries once, measure each candidate entry's firing rate, drop the entries above
a threshold. `scripts/91_domain_vocab.py --select dfcap` filtered on *lexical* document
frequency, which is a different quantity and does not predict the harm.

Writes a `dom_<corpus>_qf` vocabulary tag; evaluate it with
`scripts/92_domain_eval.py --select qf`.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening, precision
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("qffilter", "97_qf_filter.log")
DEV = "cuda"


def main(a):
    import importlib.util as iu
    spec = iu.spec_from_file_location("d92", paths.REPO / "scripts" / "92_domain_eval.py")
    d92 = iu.module_from_spec(spec); spec.loader.exec_module(d92)
    col, qids, qtexts, qrels = d92.load_domain(a.name)
    src = f"dom_{a.name}"
    z = np.load(paths.vocab_file(src))
    spec2 = iu.spec_from_file_location("e43", paths.REPO / "scripts" / "43_encode_eval.py")
    e43 = iu.module_from_spec(spec2); spec2.loader.exec_module(e43)
    cfg, head, enc, E_base, AB_base = e43.load_ckpt(a.name_model)
    layer = cfg["layer"]
    li = paths.layers_for(cfg["encoder"]).index(layer)
    pz = np.load(paths.proto_file(cfg["encoder"], src))
    dom = z["is_domain"] & (pz["cntA"] > 0)
    blob = torch.load(paths.CKPT / a.name_model / "head.pt", map_location="cpu", weights_only=False)
    mu, W = torch.as_tensor(blob["muV"]), torch.as_tensor(blob["WV"])
    Ed = F.normalize((torch.as_tensor(np.asarray(pz["protoA"][:, li][dom], np.float32)) - mu) @ W,
                     dim=-1).to(DEV)
    E = torch.cat([E_base, Ed], 0)
    nB = E_base.shape[0]
    mu_q, Wq = whitening.load(paths.ART / "whiten" / f"{cfg['encoder']}_L{layer}_Q.npz")
    tfQ = whitening.Transform("whitened", mu_q, Wq, DEV)

    g = np.random.default_rng(0)
    if a.mode == "query":
        # a query sample (query logs, in a deployment) -- no relevance labels
        sel = np.sort(g.choice(len(qtexts), size=min(a.n_queries, len(qtexts)), replace=False))
        with Timer(f"{a.name}/{cfg['encoder']}: query firing over {len(sel)} queries", lg):
            qi, qv = d92.encode([qtexts[i] for i in sel], enc, head, tfQ, E, None, layer, True, 256)
        thr = a.max_qf
    else:
        # a DOCUMENT sample of the corpus being indexed -- the thing a deployment always has.
        # This measures the entry's firing breadth under the model, which for the hub
        # entries is 100-2000x its lexical frequency (96_domain_diagnose: `predisposes`
        # occurs in 0.04% of trec-covid documents and fires on 84% of them).
        mu_h, Wh = whitening.load(paths.ART / "whiten" / f"{cfg['encoder']}_L{layer}_H.npz")
        tfH = whitening.Transform("whitened", mu_h, Wh, DEV)
        sel = np.sort(g.choice(len(col), size=min(a.n_docs, len(col)), replace=False))
        with Timer(f"{a.name}/{cfg['encoder']}: document firing over {len(sel)} passages", lg):
            qi, qv = d92.encode(col.texts(sel), enc, head, tfH, E, None, layer, False, 1024)
        thr = a.max_df
    del enc
    torch.cuda.empty_cache()
    qf = np.bincount(qi[qv > 0].ravel(), minlength=E.shape[0]).astype(np.float64) / len(sel)
    qf_ins = qf[nB:]
    if a.rel_pct is not None:
        # Relative threshold: an inserted entry may not fire more broadly than the trained
        # vocabulary's own entries do on THIS corpus under THIS backbone. Self-calibrating:
        # e5 fires everything broadly (helpful entries included), so a fixed 10% cut kept
        # only 571/2003 of its scifact entries; the decoders fire narrowly. The p-th
        # percentile of the trained entries' firing rates is the cut, no magic number.
        thr = float(np.percentile(qf[:nB][qf[:nB] > 0], a.rel_pct)) if (qf[:nB] > 0).any() else thr
        lg.info(f"relative threshold: {a.rel_pct}th percentile of trained-entry firing = {thr:.4f}")
    keep_ins = qf_ins <= thr
    lg.info(f"{a.name}/{cfg['encoder']}: inserted {a.mode}-firing mean {qf_ins.mean():.4f} "
            f"max {qf_ins.max():.3f} | keeping {int(keep_ins.sum())}/{len(qf_ins)} at <={thr}")

    words = [str(w) for w in z["words"]]
    dom_idx = np.flatnonzero(dom)
    drop_idx = dom_idx[~keep_ins]
    dropped = [words[i] for i in drop_idx]
    lg.info(f"dropped (top 20 by qf): " +
            ", ".join(f"{words[dom_idx[j]]}({qf_ins[j]:.2f})" for j in np.argsort(-qf_ins)[:20]))

    # a new tag whose is_domain keeps only the surviving entries; everything else is copied
    kind = "qf" if a.mode == "query" else ("mdfr" if a.rel_pct is not None else "mdf")
    tag = f"{src}_{kind}-{a.name_model}"
    is_dom = z["is_domain"].copy()
    is_dom[drop_idx] = False
    np.savez(paths.vocab_file(tag), **{**{k: z[k] for k in z.files}, "is_domain": is_dom})
    sp = np.load(paths.vsplits_file(src))
    dm = sp["domain"].copy(); dm[drop_idx] = False
    np.savez(paths.vsplits_file(tag), **{**{k: sp[k] for k in sp.files}, "domain": dm})
    for e in {cfg["encoder"], "e5"}:
        p = paths.proto_file(e, src)
        if p.exists() and not paths.proto_file(e, tag).exists():
            os.symlink(p, paths.proto_file(e, tag))
    save_json(dict(corpus=a.name, model=a.name_model, encoder=cfg["encoder"], mode=a.mode,
                   threshold=thr,
                   n_candidates=int(dom.sum()), n_kept=int(keep_ins.sum()),
                   n_dropped=int((~keep_ins).sum()), qf_mean_before=float(qf_ins.mean()),
                   qf_mean_after=float(qf_ins[keep_ins].mean()) if keep_ins.any() else None,
                   dropped_examples=dropped[:60]),
              paths.RESULTS / f"97_{kind}_filter_{a.name}_{a.name_model}.json")
    lg.info(f"wrote vocabulary tag {tag}: evaluate with --select {tag[len(src)+1:]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="scifact")
    ap.add_argument("--name-model", default="V1oracle")
    ap.add_argument("--max-qf", type=float, default=0.05,
                    help="drop inserted entries firing on more than this share of queries")
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--mode", default="query", choices=["query", "doc"])
    ap.add_argument("--max-df", type=float, default=0.10,
                    help="doc mode: drop inserted entries the model fires on more than this share of sampled passages")
    ap.add_argument("--n-docs", type=int, default=5000)
    ap.add_argument("--rel-pct", type=float, default=None,
                    help="doc mode: cut at this percentile of the TRAINED entries' firing rates instead of --max-df")
    main(ap.parse_args())
