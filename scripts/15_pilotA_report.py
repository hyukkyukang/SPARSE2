"""Render Pilot A's deliverable table and apply the §A.4 decision rule."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json

lg = get_logger("pilotA-rep", "15_pilotA_report.log")


def load(enc):
    p = paths.RESULTS / f"14_pilotA_{enc}_report.json"
    return json.load(open(p)) if p.exists() else None


def rows(d, enc):
    out = []
    for k, v in d.items():
        L, rep, tr = k.split("|")
        out.append(dict(
            encoder=enc, layer=int(L[1:]), rep=rep, transform=tr,
            hit10=v["content"]["self_hit10"], hit10_stop=v["stop"]["self_hit10"],
            hit10_all=v["all"]["self_hit10"],
            proto_hit10=None, auc=v["content"]["auc"],
            rel_mrr=v.get("related_mrr", float("nan")),
            rel_ratio=v.get("related_ratio", float("nan")),
            zgap=v["content"]["zgap"], pr=v["content"]["pr"],
            nnz_tok=v["content"]["nnz_tok"], tau=v["tau"],
            tau_pct=v["tau_token_percentile"],
            sense=v.get("sense", {}).get("accuracy", float("nan")),
            jaccard=v.get("splade_overlap", {}).get("jaccard", float("nan")),
            coverage=v.get("splade_overlap", {}).get("coverage", float("nan"))))
    return out


def table(rs, transform="whitened", reps=("R1-none", "R2")):
    hdr = (f"| encoder | layer | rep | cross-rep hit@10 | stop hit@10 | rel-MRR | x rand | "
           f"z-gap | PR | nnz/tok | sense | SPLADE J | cov |")
    sep = "|" + "---|" * 13
    lines = [hdr, sep]
    for r in sorted(rs, key=lambda x: (x["encoder"], x["rep"], x["layer"])):
        if r["transform"] != transform or r["rep"] not in reps:
            continue
        lines.append(
            f"| {r['encoder']} | {r['layer']} | {r['rep']} | {r['hit10']:.3f} | "
            f"{r['hit10_stop']:.3f} | {r['rel_mrr']:.4f} | {r['rel_ratio']:.0f}x | "
            f"{r['zgap']:.2f} | {r['pr']:.1f} | {r['nnz_tok']:.1f} | {r['sense']:.3f} | "
            f"{r['jaccard']:.3f} | {r['coverage']:.2f} |")
    return "\n".join(lines)


def main(encoders):
    allrows, data = [], {}
    for e in encoders:
        d = load(e)
        if d is None:
            lg.warning(f"no results for {e}"); continue
        data[e] = d
        allrows += rows(d, e)
    cand = [r for r in allrows if r["transform"] == "whitened"
            and r["rep"] == "R1-none" and r["encoder"] != "colbert"]
    cand.sort(key=lambda r: (-r["hit10"], -r["rel_mrr"]))
    ok = [r for r in cand if r["sense"] >= 0.8]
    chosen = (ok or cand)[0]
    top2 = [r["layer"] for r in
            sorted([r for r in cand if r["encoder"] == chosen["encoder"]],
                   key=lambda r: -r["hit10"])[:2]]

    # H-checks (§A.4)
    best_by_enc = {}
    for e in {r["encoder"] for r in cand}:
        rs = [r for r in cand if r["encoder"] == e]
        best_by_enc[e] = max(rs, key=lambda r: r["hit10"])
    h3 = {}
    for e in data:
        for lay in {r["layer"] for r in allrows if r["encoder"] == e}:
            g = {r["transform"]: r for r in allrows
                 if r["encoder"] == e and r["layer"] == lay and r["rep"] == "R1-none"}
            if {"raw", "whitened"} <= set(g):
                h3[f"{e}_L{lay}"] = dict(
                    zgap_ratio=g["whitened"]["zgap"] / max(g["raw"]["zgap"], 1e-9),
                    nnz_ratio=g["whitened"]["nnz_tok"] / max(g["raw"]["nnz_tok"], 1e-9))
    dec = dict(
        chosen_encoder=chosen["encoder"], chosen_layer=chosen["layer"],
        chosen_hit10=chosen["hit10"], chosen_sense=chosen["sense"],
        top2_layers=sorted(top2),
        H1_hit10_ge_0p4=bool(chosen["hit10"] >= 0.4),
        H1_rel_mrr_ge_3x=bool(chosen["rel_ratio"] >= 3),
        H2_best_layer_by_encoder={e: v["layer"] for e, v in best_by_enc.items()},
        H3_whitening=h3,
        H4_sense_ge_0p8=bool(chosen["sense"] >= 0.8),
        best_by_encoder={e: dict(layer=v["layer"], hit10=v["hit10"], sense=v["sense"])
                         for e, v in best_by_enc.items()})
    save_json(dec, paths.RESULTS / "15_pilotA_decision.json")

    md = ["# Pilot A — token-state geometry against a text-defined vocabulary", "",
          "Cross-representation self-hit@10 = probe token state vs **R1** (bare-string)",
          "entries: does a contextual state align with the embedding of the *text* that",
          "names it? This is the metric §A.4 selects on.", "",
          "## Whitened, R1 (bare string) and R2 (prototype)", "",
          table(allrows), "",
          "## Raw vs centered vs whitened (R1, best layer per encoder)", ""]
    for e in data:
        b = best_by_enc.get(e)
        if not b:
            continue
        md.append(f"**{e}, layer {b['layer']}**", )
        md.append("")
        md.append(table([r for r in allrows if r["encoder"] == e
                         and r["layer"] == b["layer"]], transform="raw",
                        reps=("R1-none", "R2")))
        for tr in ("centered", "whitened"):
            md.append(table([r for r in allrows if r["encoder"] == e
                             and r["layer"] == b["layer"]], transform=tr,
                            reps=("R1-none", "R2")).split("\n", 2)[2])
        md.append("")
    md += ["## Decision (§A.4)", "", "```json", json.dumps(dec, indent=2), "```"]
    (paths.REPORTS / "pilotA.md").write_text("\n".join(md))
    lg.info(f"chosen: {chosen['encoder']} layer {chosen['layer']} "
            f"hit10={chosen['hit10']:.3f} sense={chosen['sense']:.3f}; top2={sorted(top2)}")
    return dec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="e5,bge,colbert")
    main(ap.parse_args().encoders.split(","))
