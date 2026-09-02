"""Render Pilot B's deliverable and apply the §B.4 decision rule."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json

lg = get_logger("pilotB-rep", "21_pilotB_report.log")


def main(files):
    data = {}
    for f in files:
        p = paths.RESULTS / f
        if p.exists():
            d = json.load(open(p))
            data[f"{d['encoder']}_L{d['layer']}"] = d
    md = ["# Pilot B — bare string vs contextual prototype", ""]
    dec = {}
    for key, d in data.items():
        reps = list(d["hubness"])
        md += [f"## {key}", "",
               "| rep | hub skew | hub share | Gini | Spearman df~freq | runaway % | "
               "dead % | nnz/doc | in-context hit@10 | self-act rank | P(rank1) | sense |",
               "|" + "---|" * 12]
        for r in reps:
            h, f_ = d["hubness"][r], d["df"][r]
            ic = d["in_context"][r]
            md.append(
                f"| {r} | {h['skew']:.1f} | {h['hub_share']:.3f} | {h['gini']:.3f} | "
                f"{f_['spearman_df_freq']:.3f} | {100*f_['runaway_rate']:.2f} | "
                f"{100*f_['dead_rate']:.2f} | {f_['nnz_per_doc']:.0f} | "
                f"{ic['content']['self_hit10']:.3f} | "
                f"{ic['self_activation']['median_rank']:.0f} | "
                f"{ic['self_activation']['p_rank1']:.3f} | {ic['sense']['accuracy']:.3f} |")
        md += ["", "### 20 largest hubs", ""]
        for r in reps:
            md.append(f"* **{r}**: " + ", ".join(
                f"`{w}`({c})" for w, c, _, _ in d["hubness"][r]["top20_hubs"]))
        md += ["", "### Stability curve (R2, disjoint occurrence sets)", "",
               "| k | " + " | ".join(str(k) for k in sorted(map(int, d["stability"]))) + " |",
               "|" + "---|" * (len(d["stability"]) + 1)]
        ks = sorted(map(int, d["stability"]))
        md.append("| Jaccard@50 | " + " | ".join(
            f"{d['stability'][str(k)]['jaccard']:.3f}" for k in ks) + " |")
        md.append("| whitened cos | " + " | ".join(
            f"{d['stability'][str(k)]['whitened_cos']:.3f}" for k in ks) + " |")
        md += ["", f"**k\\*** (smallest k reaching 0.8 x the k=50 Jaccard) = {d['k_star']}", ""]

        # §B.4 decision
        cands = [r for r in reps if not r.endswith("-shared")]
        best_ic = max(d["in_context"][r]["content"]["self_hit10"] for r in cands)
        elig = [r for r in cands
                if d["in_context"][r]["content"]["self_hit10"] >= best_ic - 0.02]
        pick = min(elig, key=lambda r: d["hubness"][r]["skew"])
        h5 = dict(
            hub_skew_ratio=d["hubness"]["R1"]["skew"] / max(d["hubness"]["R2"]["skew"], 1e-9),
            spearman_R1=d["df"]["R1"]["spearman_df_freq"],
            spearman_R2=d["df"]["R2"]["spearman_df_freq"],
            in_context_R1=d["in_context"]["R1"]["content"]["self_hit10"],
            in_context_R2=d["in_context"]["R2"]["content"]["self_hit10"])
        h7 = dict(own=d["in_context"]["R1"]["content"]["self_hit10"],
                  shared=d["in_context"].get("R1-shared", {}).get(
                      "content", {}).get("self_hit10"))
        dec[key] = dict(chosen_rep=pick, k_star=d["k_star"], H5=h5, H7=h7,
                        runaway_max=max(d["df"][r]["runaway_rate"] for r in cands),
                        pilotE_mandatory=bool(
                            all(d["df"][r]["runaway_rate"] > 0.05 for r in cands)))
        md += ["### Decision (§B.4)", "", "```json", json.dumps(dec[key], indent=2), "```", ""]
    save_json(dec, paths.RESULTS / "21_pilotB_decision.json")
    (paths.REPORTS / "pilotB.md").write_text("\n".join(md))
    lg.info(json.dumps(dec, indent=2)[:1500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    main(ap.parse_args().files)
