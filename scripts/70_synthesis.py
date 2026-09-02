"""Assemble every pilot's decision into one findings document."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger

lg = get_logger("synth", "70_synthesis.log")


def jload(name):
    p = paths.RESULTS / name
    return json.load(open(p)) if p.exists() else None


def main():
    A = jload("15_pilotA_decision.json")
    B = jload("21_pilotB_decision.json")
    C = jload("32_pilotC_decision.json")
    D = jload("46_pilotD_decision.json")
    E = jload("51_pilotE_decision.json")
    refs_full = jload("03_reference_full.json") or {}
    refs_c1 = jload("06_references_c1.json") or {}
    dense_full = jload("04_dense_full.json") or {}
    vocab = jload("01_vocab.json") or {}
    sense = jload("17_sense_verification.json") or {}
    full = jload("60_full_corpus.json")

    md = ["# Dynamic-vocabulary LSR — pilot study findings", "",
          "Everything below comes from `results/*.json`; each pilot's own report in",
          "`reports/` carries the full tables.", "",
          "## Pipeline validation (§Appendix, run before Pilot A)", "",
          "| system | full collection MRR@10 | published |", "|---|---|---|"]
    if "bm25" in refs_full:
        md.append(f"| BM25 (k1=0.82, b=0.68) | {refs_full['bm25']['mrr@10']:.4f} | 0.1875 |")
    if "spladepp" in refs_full:
        md.append(f"| SPLADE++ CoCondenser-ED | {refs_full['spladepp']['mrr@10']:.4f} | 0.383 |")
    for k, v in dense_full.items():
        md.append(f"| {k}-base (ours) | {v['mrr@10']:.4f} | ~0.35 |")
    md += ["", f"Vocabulary V: {vocab.get('size')} entries, "
           f"{vocab.get('n_qualified')} types cleared the {vocab.get('min_occ')}-occurrence "
           f"floor, bottom-decile minimum frequency {vocab.get('freq_min')}.", ""]

    def block(title, obj, keys):
        if not obj:
            return [f"## {title}", "", "_not run_", ""]
        out = [f"## {title}", ""]
        for k in keys:
            if k in obj:
                out.append(f"* **{k}**: `{json.dumps(obj[k])}`")
        return out + [""]

    md += block("Pilot A — token-state geometry", A,
                ["chosen_encoder", "chosen_layer", "chosen_hit10", "top2_layers",
                 "H1_verdict", "H2_verdict", "H2_best_layer_by_encoder", "H3_verdict",
                 "H4_sense_max", "H4_sense_constraint_met", "literal_rule_pick",
                 "selection_conflict", "metric_peaks_by_layer"])
    if B:
        for k, v in B.items():
            md += block(f"Pilot B — entry representation ({k})", v,
                        ["chosen_rep", "chosen_rep_by_skew", "hub_share", "hub_skew",
                         "k_star", "H5_verdict", "H6_k10_ratio", "H6_verdict",
                         "H7_verdict", "runaway_max", "pilotE_mandatory"])
    md += block("Pilot C — training-free retrieval on C1", C,
                ["verdict", "ratio_to_bm25", "k_d", "k_q", "nnz_d", "nnz_q",
                 "saturation", "tau", "include_lora_arm"])
    if refs_c1:
        md += ["### References on C1", "",
               "| system | MRR@10 | R@100 | R@1000 |", "|---|---|---|---|"]
        for k, v in refs_c1.items():
            md.append(f"| {k} | {v['mrr@10']:.4f} | {v['r@100']:.4f} | {v['r@1000']:.4f} |")
        md.append("")
    md += block("Pilot D — held-out vocabulary generalisation", D,
                ["recipe", "H10_verdict", "H11_verdict", "H12_verdict", "H13_verdict",
                 "H14_verdict", "pilotE_triggered", "summary"])
    md += block("Pilot E — insertion-time calibration", E,
                ["H15_verdict", "chosen_variant", "rho", "verdict"])
    if full:
        md += ["## Full-corpus confirmation (8.84M passages)", "",
               "| system | MRR@10 | R@100 | R@1000 |", "|---|---|---|---|"]
        for k, v in full.items():
            md.append(f"| {k} | {v['mrr@10']:.4f} | {v['r@100']:.4f} | {v['r@1000']:.4f} |")
        md.append("")
    md += ["## Measurement caveats", "",
           f"* Sense labels: manual verification agreement "
           f"{sense.get('round2', {}).get('agreement_strict', float('nan')):.3f} after "
           f"correcting {len(sense.get('dropped_indicators', {}))} indicator lists "
           f"(was {sense.get('round1', {}).get('agreement_strict', float('nan')):.3f}).",
           "* C1 contains every reference system's own candidates but not ours, so C1",
           "  numbers are mildly optimistic for us; the full-corpus run is the headline.",
           "* See `notes/deviations.md` for every departure from the written protocol.", ""]
    (paths.REPORTS / "FINDINGS.md").write_text("\n".join(md))
    lg.info("wrote reports/FINDINGS.md")


if __name__ == "__main__":
    main()
