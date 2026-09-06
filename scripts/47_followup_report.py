"""Follow-up report: the questions left open by the pilots, answered from the rebuild.

Reads whatever exists in RESULTS (the report can be regenerated at any point of the
pipeline) and writes REPORTS/FOLLOWUPS.md plus RESULTS/47_followups.json.

Sections
  1. setup on this machine (C1, references)
  2. Pilot C at layer 12: contextual prototypes vs bare-string entries
  3. cluster split: entry-wise vs cluster-wise vs meta-calibrated vocabulary dropout, linear head
  4. Pilot E on the cluster model
  5. random split: baselines, seeds, capacity, distillation ceiling, controls
  6. rare split
  7. phrase / entity insertion
  8. encoder-training arms
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json

lg = get_logger("followups", "47_followup_report.log")
LOG15 = float(np.log(1.5))

LABEL = {
    "V1": "V1 (frozen, no dropout)", "V1_s2": "V1, seed 2",
    "V1vd": "V1 + entry-wise VD", "V1vd_s2": "V1 + entry-wise VD, seed 2",
    "V1cvd": "V1 + cluster-wise VD", "V1meta": "V1 + cluster VD + meta calibration",
    "V1lin": "V1, linear head", "V1dist": "V1 + dense-teacher distillation",
    "Crand": "C-rand (random vocabulary)",
    "V1_cluster": "V1 (frozen, no dropout)", "V1vd_cluster": "V1 + entry-wise VD",
    "V1cvd_cluster": "V1 + cluster-wise VD", "V1meta_cluster": "V1 + cluster VD + meta calibration",
    "V1lin_cluster": "V1, linear head", "V3_cluster": "V3 (full fine-tuning)",
    "V1_rare": "V1 (frozen, no dropout)", "V1vd_rare": "V1 + entry-wise VD",
    "V1_phrase": "V1oracle + 2,000 inserted phrases",
    "V1_R1": "V1, bare-string entries", "V1_cluster_R1": "V1, bare-string entries",
    "V1cvd_cluster_R1": "V1, bare-string entries + cluster-wise VD",
    "V1_cluster_Wh": "V1, entries whitened with the token-side transform",
    "V1_Wh": "V1, entries whitened with the token-side transform",
    "V1norm": "V1 + tail normalisation",
    "V1norm_rare": "V1 + tail normalisation",
    "V1norm_cluster_R1": "V1, bare-string entries + tail normalisation",
    "V1param": "V1, trained entry rows (parameterized-vocabulary control)",
    "V1eh": "V1 + shared entry-side map", "V1eh_cluster": "V1 + shared entry-side map",
    "V1ehnorm_cluster": "V1 + entry-side map + tail normalisation",
    "V1k100_cluster": "V1, k=100 prototypes",
    "V1_phrase_norm": "V1oracle + 2,000 phrases + tail normalisation",
    "V1norm_cluster": "V1 + tail normalisation",
    "V1_cluster_s2": "V1, seed 2", "V1norm_cluster_s2": "V1 + tail normalisation, seed 2",
    "V1normz_cluster": "V1 + moment normalisation (contrast)",
    "V1normWh_cluster": "V1 + tail normalisation + token-side entry transform",
    "V2": "V2 (LoRA)", "V2vd": "V2 (LoRA) + VD", "V3": "V3 (full fine-tuning)", "V3vd": "V3 + VD",
}


def load(name):
    p = paths.RESULTS / name
    return json.load(open(p)) if p.exists() else None


def d_row(name):
    d = load(f"44_pilotD_{name}.json")
    if d is None:
        return None
    qh, lex, act = d.get("Q_H", {}), d.get("Q_H_lex", {}), d.get("activation", {})
    va = qh.get("vs_alias")
    r = dict(name=name, label=LABEL.get(name, name), oracle=d.get("oracle"),
             mrr_all=d["overall_all"]["mrr@10"], mrr_seen=d["overall_seen"]["mrr@10"],
             r100_all=d["overall_all"]["r@100"],
             nnz_d=d["nnz_d"], nnz_q=d["nnz_q"],
             n_QH=qh.get("n"), n_lex=qh.get("n_lex", (lex or {}).get("n")),
             rho=(qh.get("rho") or {}).get("rho"), rho_lo=(qh.get("rho") or {}).get("lo"),
             rho_hi=(qh.get("rho") or {}).get("hi"), den=(qh.get("denominator") or {}).get("diff"),
             den_valid=qh.get("denominator_valid"), gap_oracle=qh.get("gap_to_oracle_rel"),
             rho_lex=((lex or {}).get("rho") or {}).get("rho"),
             alias=qh.get("alias_mrr"), beats_alias=(va["excludes_zero"] and va["diff"] > 0) if va else None,
             gap_r=act.get("signed_gap_r"), abs_gap_r=act.get("abs_gap_r"), gap_w=act.get("gap_w"),
             dead_held=act.get("dead_held"))
    return r


def fmt(x, nd=3, pct=False):
    if x is None:
        return "—"
    if isinstance(x, float) and x != x:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.{nd}f}"


def table(names, cols):
    rows = [d_row(n) for n in names]
    rows = [r for r in rows if r]
    if not rows:
        return ["_not evaluated yet_", ""]
    head = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "---|" * len(cols)
    out = [head, sep]
    for r in rows:
        out.append("| " + " | ".join(c[1](r) for c in cols) + " |")
    return out + [""]


COLS_MAIN = [
    ("arm", lambda r: r["label"]),
    ("MRR@10 all", lambda r: fmt(r["mrr_all"], 4)),
    ("seen-only", lambda r: fmt(r["mrr_seen"], 4)),
    ("rho (Q_H)", lambda r: fmt(r["rho"]) + (f" [{r['rho_lo']:.2f}, {r['rho_hi']:.2f}]" if r["rho"] is not None else "")),
    ("rho (lex)", lambda r: fmt(r["rho_lex"])),
    ("gap to oracle", lambda r: fmt(r["gap_oracle"])),
    ("signed gap_r", lambda r: fmt(r["gap_r"])),
    ("abs gap_r", lambda r: fmt(r["abs_gap_r"])),
    ("C-alias MRR", lambda r: fmt(r["alias"], 4) + ("" if r["beats_alias"] is None else (" ✓" if r["beats_alias"] else " ✗"))),
    ("nnz d/q", lambda r: f"{r['nnz_d']:.0f}/{r['nnz_q']:.0f}"),
    ("n(Q_H)", lambda r: f"{r['n_QH']}"),
]


def oracle_overall(name, split):
    """All-entries metrics of an oracle from its cached evaluation (44_pilotD_eval writes
    RUNS/D_eval_<name>_<split>.npz with the result dict inside)."""
    p = paths.RUNS / f"D_eval_{name}_{split}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return json.loads(str(z["res"])).get("overall_all")


def hist_stats(name):
    p = paths.CKPT / name / "history.json"
    if not p.exists():
        return None
    h = json.load(open(p))
    if not h:
        return None
    tail = h[-10:]
    return dict(steps=h[-1]["step"], ce=float(np.mean([x["ce"] for x in tail])),
                acc=float(np.mean([x["acc"] for x in tail])),
                nnz_d=float(np.mean([x["nnz_d"] for x in tail])),
                cal=float(np.mean([x.get("cal", 0) for x in tail])),
                kl=float(np.mean([x.get("kl", 0) for x in tail])),
                t=tail[-1]["t"], b=tail[-1]["b"])


def main():
    md = ["# Follow-up experiments — what the pilots left open", "",
          "Every number here comes from the rebuild on dslab-gpu10 (notes/deviations.md D12), "
          "so arms are compared to baselines re-run in the same rebuild, never to the A100 "
          "numbers. rho is the recovery ratio of §D.6 (share of the oracle's held-out benefit "
          "that survives insertion); signed gap_r > 0 means held-out entries over-fire relative "
          "to seen entries of the same frequency decile; |gap_r| <= log 1.5 = 0.405 is the H10 "
          "calibration bound.", ""]
    summary = {}

    # ---- 1. setup ----
    refs = load("06_references_c1.json") or {}
    c1 = load("05_c1.json") or {}
    md += ["## 1. Setup on this machine", ""]
    if c1:
        md += [f"C1 = {c1.get('union', 0):,} passages "
               f"(qrels {c1.get('qrels', 0):,}, BM25 top-100 {c1.get('bm25_full_top100', 0):,}, "
               f"SPLADE++ top-100 {c1.get('spladepp_full_top100', 0):,}, random {c1.get('random', 0):,}; "
               f"no dense candidates — D12).", ""]
    if refs:
        md += ["| reference on C1 | MRR@10 | R@100 | R@1000 |", "|---|---|---|---|"]
        for k, v in refs.items():
            md.append(f"| {k} | {v['mrr@10']:.4f} | {v['r@100']:.4f} | {v['r@1000']:.4f} |")
        md.append("")
    summary["refs"] = {k: v["mrr@10"] for k, v in refs.items()}

    # ---- 2. Pilot C at layer 12: R2 vs R1 ----
    md += ["## 2. Untrained retrieval at layer 12: contextual prototypes vs bare-string entries", "",
           "Pilot C never ran bare-string (R1) entries at the layer that retrieves best; if R1 "
           "were close to R2 here, insertion would cost one forward pass instead of ~50 occurrences.", ""]
    pc = {}
    for rep in ("R2", "R1"):
        d = load(f"31_pilotC_e5_L12_{rep}.json")
        if d:
            pc[rep] = d
    if pc:
        md += ["| entries | best cell | MRR@10 | 95% CI | R@100 | R@1000 | vs BM25 |", "|---|---|---|---|---|---|---|"]
        bm = (refs.get("bm25") or {}).get("mrr@10")
        for rep, d in pc.items():
            b = d["best"]; ci = b["mrr@10_ci"]
            md.append(f"| {'R2 contextual prototype' if rep == 'R2' else 'R1 bare string'} | {b['name']} | "
                      f"{b['mrr@10']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {b['r@100']:.4f} | {b['r@1000']:.4f} | "
                      f"{(b['mrr@10'] / bm):.2f}x |" if bm else "")
            summary[f"pilotC_L12_{rep}"] = b["mrr@10"]
        if "R1" in pc and "R2" in pc:
            r = pc["R1"]["best"]["mrr@10"] / pc["R2"]["best"]["mrr@10"]
            md += ["", f"Bare-string entries reach **{r:.2f}x** the prototype MRR@10 at layer 12 "
                   f"(at layer 9 the pilot found 0.65x). " +
                   ("They remain clearly worse: insertion keeps needing occurrences." if r < 0.9 else
                    "They are close: one-forward-pass insertion is back on the table.")]
    else:
        md.append("_not run yet_")
    md.append("")

    # ---- 3. cluster split ----
    md += ["## 3. Cluster split: does semantic vocabulary dropout fix over-firing?", "",
           "The pilot's one clear negative: entries from an unseen *region* over-fire "
           "(signed gap_r +0.47 on the A100 run) and recover only 57% of the oracle's benefit. "
           "Entry-wise dropout cannot address it (every dropped entry keeps trained neighbours); "
           "cluster-wise dropout removes whole regions per step; the meta arm additionally keeps "
           "some regions out of the ranking loss for the whole run and trains their activation "
           "rate to match trained entries (a training-time gap_r). The linear head is the "
           "capacity control: if it cannot over-fire, co-adaptation of the MLP is the cause.", ""]
    cl = ["V1_cluster", "V1vd_cluster", "V1cvd_cluster", "V1meta_cluster", "V1lin_cluster",
          "V1_cluster_R1", "V1cvd_cluster_R1", "V1_cluster_Wh",
          "V1eh_cluster", "V1ehnorm_cluster", "V1k100_cluster", "V1norm_cluster", "V1_cluster_s2", "V1norm_cluster_s2", "V1normz_cluster",
          "V1normWh_cluster", "V1norm_cluster_R1", "V3_cluster"]
    md += table(cl, COLS_MAIN)
    rows = {n: d_row(n) for n in cl}
    rows = {n: r for n, r in rows.items() if r}
    if rows:
        best_cal = min(rows.items(), key=lambda kv: abs(kv[1]["abs_gap_r"] or 9))
        best_rho = max(rows.items(), key=lambda kv: kv[1]["rho"] or -9)
        md += [f"Smallest |gap_r|: **{best_cal[1]['label']}** ({best_cal[1]['abs_gap_r']:.3f}); "
               f"highest rho: **{best_rho[1]['label']}** ({best_rho[1]['rho']:.3f}).", ""]
        summary["cluster"] = {n: dict(rho=r["rho"], gap_r=r["gap_r"], mrr=r["mrr_all"]) for n, r in rows.items()}
    hs = {n: hist_stats(n) for n in cl}
    hs = {n: h for n, h in hs.items() if h}
    if hs:
        md += ["Training tail (mean of the last 10 logged steps):", "",
               "| arm | steps | CE | in-batch acc | nnz(d) | calibration loss | t | b |", "|---|---|---|---|---|---|---|---|"]
        for n, h in hs.items():
            md.append(f"| {LABEL.get(n, n)} | {h['steps']} | {h['ce']:.3f} | {h['acc']:.3f} | {h['nnz_d']:.0f} | "
                      f"{h['cal']:.4f} | {h['t']:.1f} | {h['b']:.2f} |")
        md.append("")

    # ---- 4. Pilot E ----
    md += ["## 4. Pilot E: insertion-time calibration on the cluster model", ""]
    pe = load("51_pilotE_decision.json")
    if pe:
        md += ["| variant | rho | 95% CI | gap to oracle | signed gap_r | gap_w | nnz(d) | Δnnz(d) |", "|" + "---|" * 8]
        for v, r in pe["rows"].items():
            md.append(f"| {v} | {fmt(r['rho'])} | " +
                      (f"[{r['rho_lo']:.2f}, {r['rho_hi']:.2f}]" if r.get("rho_lo") is not None else "—") +
                      f" | {fmt(r['gap_to_oracle'])} | {fmt(r['signed_gap_r'], 4)} | {fmt(r['gap_w'], 4)} | "
                      f"{r['nnz_d']:.0f} | {r['delta_nnz_d']:+.0f} |")
        md += ["", f"Verdict: {pe['verdict']} (H15: {pe['H15_verdict']})", ""]
        summary["pilotE"] = {v: r["rho"] for v, r in pe["rows"].items()}
    else:
        md += ["_not run yet_", ""]

    # ---- 5. random split ----
    md += ["## 5. Random split: baselines, seeds, capacity, distillation, controls", ""]
    rn = ["V1", "V1_s2", "V1vd", "V1vd_s2", "V1cvd", "V1meta", "V1lin", "V1dist", "V1_R1",
          "V1_Wh", "V1norm", "V1eh", "V1param", "Crand"]
    md += table(rn, COLS_MAIN)
    o1 = load("44_pilotD_V1.json")
    dense = (refs.get("dense_e5") or {}).get("mrr@10")
    ceil = [(lab, oracle_overall(n, "random")) for n, lab in
            (("V1oracle", "V1 oracle (frozen encoder, all entries)"),
             ("V1dist_oracle", "V1 oracle + distillation from the dense score"),
             ("V3oracle", "V3 oracle (full fine-tuning, all entries)"))]
    ceil = [(lab, m) for lab, m in ceil if m]
    if ceil:
        md += ["Effectiveness ceiling (all entries, C1) — the fork between a pure text-defined "
               "vocabulary and a hybrid extension of a fixed LSR model turns on this table:", "",
               "| model | MRR@10 | R@100 |", "|---|---|---|"]
        for lab, m in ceil:
            md.append(f"| {lab} | {m['mrr@10']:.4f} | {m['r@100']:.4f} |")
        for k, lab in (("bm25", "BM25"), ("dense_e5", "dense e5 (the teacher)"), ("spladepp", "SPLADE++")):
            if refs.get(k):
                md.append(f"| {lab} | {refs[k]['mrr@10']:.4f} | {refs[k]['r@100']:.4f} |")
        md.append("")
        summary["ceiling"] = {lab: m["mrr@10"] for lab, m in ceil}

    # ---- 6. rare ----
    md += ["## 6. Rare split", ""]
    md += table(["V1_rare", "V1vd_rare", "V1norm_rare"], COLS_MAIN)

    # ---- 7. phrases ----
    md += ["## 7. Phrase / entity insertion", ""]
    pv = load("80_phrase_vocab.json")
    if pv:
        md += [f"{pv['n_phrases']:,} Title-case bigrams with >= {paths.V_MIN_OCC} passages in P "
               f"(of {pv['n_qualified']:,} qualifying; collection frequency {pv['freq_min']:,}–{pv['freq_max']:,}). "
               f"Examples: {', '.join(pv['examples'][:15])}.", ""]
    md += table(["V1_phrase", "V1_phrase_norm"], COLS_MAIN)
    ph = load("44_pilotD_V1_phrase.json")
    ow = oracle_overall("V1oracle", "random")
    op = oracle_overall("V1oracle_phrase", "phrase")
    if ph and ow:
        md += ["| model | MRR@10 all entries |", "|---|---|",
               f"| word-only oracle (V1oracle, 30k words) | {ow['mrr@10']:.4f} |",
               f"| V1oracle + phrases inserted, no retraining | {ph['overall_all']['mrr@10']:.4f} |"]
        if op:
            md.append(f"| oracle trained with words + phrases | {op['mrr@10']:.4f} |")
        md += ["", "Read these three with the Q_H numbers above, not instead of them. Overall "
               "MRR@10 barely moves because the phrases matter to only ~13% of dev queries; on "
               "the Q_H subset where they do matter, the retrained oracle gains 0.042 "
               "(0.2645 vs 0.2222 with them zeroed). So the phrases *are* useful entries when "
               "trained in, and post-hoc insertion is what fails to capture that.", ""]

    # ---- 8. encoder arms ----
    md += ["## 8. Encoder-training arms", ""]
    md += table(["V2", "V2vd", "V3", "V3vd"], COLS_MAIN)

    dec = load("46_pilotD_decision.json")
    if dec:
        md += ["## Pilot D decision rule on the rebuild", "", "```json",
               json.dumps(dec["decision"], indent=2), "```", ""]
    (paths.REPORTS / "FOLLOWUPS.md").write_text("\n".join(md))
    save_json(summary, paths.RESULTS / "47_followups.json")
    lg.info(f"wrote {paths.REPORTS / 'FOLLOWUPS.md'}")


if __name__ == "__main__":
    main()
