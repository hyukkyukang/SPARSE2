"""Pilot C report and §C.4 decision: does the untrained projection retrieve, and
which operating point does Pilot D initialise from?"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json

lg = get_logger("pilotC-dec", "32_pilotC_decision.log")


def main(f, r1_prefix):
    d = json.load(open(paths.RESULTS / f))
    refs = json.load(open(paths.RESULTS / "06_references_c1.json"))
    bm25 = refs["bm25"]["mrr@10"]
    dense = (refs.get(f"dense_{d['encoder']}") or refs.get("dense_e5") or {}).get("mrr@10", float("nan"))
    splade = refs["spladepp"]["mrr@10"]
    best = d["best"]
    ratio = best["mrr@10"] / bm25
    verdict = ("pass" if ratio >= 0.8 else
               "weak" if ratio >= 0.4 else "fail")
    sp = d.get("score_preservation", {})
    if verdict == "fail" and sp.get("spearman_mean", 1) >= 0.3:
        verdict = "weak"
    cfg = dict(encoder=d["encoder"], layer=d["layer"], rep=d["rep"],
               r1_prefix=r1_prefix,
               tau=d["tau_d"][str(d["best_cfg"]["nnz_d"])]["tau"],
               tau_q=d["tau_q"][str(d["best_cfg"]["nnz_q"])]["tau"],
               k_d=d["best_cfg"]["k_d"], k_q=d["best_cfg"]["k_q"],
               saturation=d["best_cfg"]["saturation"],
               nnz_d=d["best_cfg"]["nnz_d"], nnz_q=d["best_cfg"]["nnz_q"],
               lam_d=3e-4, lam_q=9e-4,
               verdict=verdict, ratio_to_bm25=ratio,
               include_lora_arm=bool(verdict == "weak"))
    save_json(cfg, paths.RESULTS / "32_pilotC_decision.json")

    md = ["# Pilot C — training-free end-to-end retrieval on C1", "",
          f"Encoder **{d['encoder']}**, layer **{d['layer']}**, entry representation "
          f"**{d['rep']}**; C1 = {d['n_docs']:,} passages, dev-small queries.", "",
          "## References on C1", "",
          "| system | MRR@10 | R@100 | R@1000 |", "|---|---|---|---|"]
    for k, v in refs.items():
        md.append(f"| {k} | {v['mrr@10']:.4f} | {v['r@100']:.4f} | {v['r@1000']:.4f} |")
    md += ["", "## Grid (36 cells, all re-truncations of one stored profile set)", "",
           "| cell | MRR@10 | 95% CI | R@100 | R@1000 |", "|---|---|---|---|---|"]
    for name, m in sorted(d["grid"].items(), key=lambda x: -x[1]["mrr@10"]):
        ci = m["mrr@10_ci"]
        md.append(f"| {name} | {m['mrr@10']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
                  f"{m['r@100']:.4f} | {m['r@1000']:.4f} |")
    md += ["", "## Operating point and sparsity", "",
           "| target nnz | tau | mean nnz (S) | token percentile |", "|---|---|---|---|"]
    for t, v in d["tau_d"].items():
        md.append(f"| doc {t} | {v['tau']:.4f} | {v['mean_nnz']:.1f} | "
                  f"{v['token_percentile']:.4f} |")
    for t, v in d["tau_q"].items():
        md.append(f"| query {t} | {v['tau']:.4f} | {v['mean_nnz']:.1f} | — |")
    if sp:
        md += ["", "## Score preservation vs the dense encoder", "",
               f"* Spearman over the dense top-100 within C1: "
               f"mean {sp['spearman_mean']:.3f}, median {sp['spearman_median']:.3f}",
               f"* Jaccard of top-100 with dense: {sp['jaccard_top100']:.3f}"]
    if "splade_overlap_truncated" in d:
        o = d["splade_overlap_truncated"]
        md += ["", f"* SPLADE++ top-20 Jaccard on the truncated representation: "
                   f"{o['jaccard']:.3f} (coverage {o['coverage']:.2f})"]
    md += ["", "## Decision (§C.4)", "",
           f"* best cell **{best['name']}**: MRR@10 = {best['mrr@10']:.4f}",
           f"* BM25 on C1 = {bm25:.4f}; ratio = **{ratio:.2f}x** "
           f"(H8 wants >= 0.80x)",
           f"* dense = {dense:.4f}, SPLADE++ = {splade:.4f}; "
           f"is the untrained run between BM25 and dense? "
           f"{'yes' if bm25 <= best['mrr@10'] <= dense else 'no'}",
           f"* verdict: **{verdict}**"
           + ("" if verdict != "weak" else " — Pilot D includes the LoRA arm from the start"),
           "", "```json", json.dumps(cfg, indent=2), "```"]
    (paths.REPORTS / "pilotC.md").write_text("\n".join(md))
    lg.info(f"verdict={verdict} ratio_to_bm25={ratio:.3f} best={best['name']} "
            f"MRR@10={best['mrr@10']:.4f}")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--r1-prefix", default="none")
    a = ap.parse_args()
    main(a.file, a.r1_prefix)
