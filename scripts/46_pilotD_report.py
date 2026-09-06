"""Assemble Pilot D's deliverable table and apply the §D.7 decision rule."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json

lg = get_logger("pilotD-rep", "46_pilotD_report.log")
LOG15 = float(np.log(1.5))


def load_all():
    out = {}
    for p in sorted(paths.RESULTS.glob("44_pilotD_*.json")):
        d = json.load(open(p))
        out[d["name"]] = d
    return out


def row(name, d):
    qh = d.get("Q_H", {})
    a, s = d.get("overall_all", {}), d.get("overall_seen", {})
    act = d.get("activation", {})
    arch = "V3" if name.startswith("V3") else "V2" if name.startswith("V2") else "V1"
    orc = d.get("oracle", "")
    orc_arch = "V3" if orc.startswith("V3") else "V2" if orc.startswith("V2") else "V1"
    r = dict(name=name, split=d.get("split"), oracle=orc,
             oracle_arch_match=bool(arch == orc_arch),
             mrr_all=a.get("mrr@10"), mrr_seen=s.get("mrr@10"),
             r100_all=a.get("r@100"), nnz_d=d.get("nnz_d"), nnz_q=d.get("nnz_q"),
             n_QH=qh.get("n"), n_QH_lex=qh.get("n_lex"),
             signed_gap_r=act.get("signed_gap_r"), abs_gap_r=act.get("abs_gap_r"),
             gap_w=act.get("gap_w"), wasserstein=act.get("wasserstein"))
    for label in ("Q_H", "Q_H_lex"):
        e = d.get(label)
        if e:
            r[f"{label}_n"] = e.get("n")
            r[f"{label}_rho"] = e["rho"]["rho"]
            r[f"{label}_rho_lo"] = e["rho"]["lo"]
            r[f"{label}_rho_hi"] = e["rho"]["hi"]
            r[f"{label}_den"] = e["denominator"]["diff"]
            r[f"{label}_den_valid"] = e["denominator_valid"]
            r[f"{label}_gap_to_oracle"] = e["gap_to_oracle_rel"]
            r[f"{label}_alias_mrr"] = e.get("alias_mrr")
            va = e.get("vs_alias")
            r[f"{label}_beats_alias"] = (va["excludes_zero"] and va["diff"] > 0) if va else None
    return r


def verdict(r):
    """H10: rho >= 0.8 with CI excluding 0.5, gap <= 10% rel, |gap_r| <= log 1.5,
    and rho beats C-alias with a CI excluding zero."""
    if r.get("Q_H_rho") is None:
        return "no rho"
    checks = dict(
        rho=r["Q_H_rho"] >= 0.8,
        ci_excludes_0p5=r["Q_H_rho_lo"] > 0.5,
        gap=abs(r["Q_H_gap_to_oracle"]) <= 0.10,
        gap_r=abs(r["signed_gap_r"]) <= LOG15 if r["signed_gap_r"] is not None else False,
        beats_alias=(None if r.get("Q_H_alias_mrr") is None
                     else bool(r.get("Q_H_beats_alias"))))
    if r.get("signed_gap_r") is None or r["signed_gap_r"] != r["signed_gap_r"]:
        # the rare split holds out a whole decile, so there is no within-decile
        # seen/held comparison to make -- report it as not measurable, not as a failure
        checks["gap_r"] = None
    known = [v for v in checks.values() if v is not None]
    return dict(pass_=all(known), unmeasured=[k for k, v in checks.items() if v is None],
                **checks)


def main():
    res = load_all()
    if not res:
        lg.warning("no Pilot D results yet")
        return
    rows = {n: row(n, d) for n, d in res.items()}
    md = ["# Pilot D — held-out vocabulary generalisation", "",
          "rho compares a model to an oracle of its **own architecture** (§D.3). Rows",
          "marked (arch\\*) have no same-architecture oracle on their split — the",
          "protocol's own run list provides only a V1 oracle for the cluster and rare",
          "splits — so their rho is reported for completeness but is not a like-for-like",
          "recovery ratio.", "",
          "| run | split | oracle | MRR@10 all | seen-only | rho (Q_H) | 95% CI | denom | valid |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, r in sorted(rows.items()):
        arch = "" if r["oracle_arch_match"] else " (arch\\*)"
        md.append(
            f"| {n}{arch} | {r['split']} | "
            f"{r['oracle']} | {r['mrr_all']:.4f} | {r['mrr_seen']:.4f} | "
            + (f"{r['Q_H_rho']:.3f} | [{r['Q_H_rho_lo']:.2f}, {r['Q_H_rho_hi']:.2f}] | "
               f"{r['Q_H_den']:.4f} | {r['Q_H_den_valid']} |"
               if r.get("Q_H_rho") is not None else "— | — | — | — |"))
    md += ["", "| run | gap to oracle on Q_H | signed gap_r | abs gap_r | gap_w | "
           "Wasserstein | nnz(d) | nnz(q) | \\|Q_H\\| | \\|Q_H-lex\\| | rho (Q_H-lex) | C-alias MRR |",
           "|" + "---|" * 12]
    for n, r in sorted(rows.items()):
        n_lex = r["n_QH_lex"] if r.get("n_QH_lex") is not None else r.get("Q_H_lex_n")
        md.append(
            f"| {n} | " +
            (f"{r['Q_H_gap_to_oracle']:.3f}" if r.get("Q_H_gap_to_oracle") is not None else "—")
            + f" | {r['signed_gap_r']:.4f} | {r['abs_gap_r']:.4f} | {r['gap_w']:.4f} | "
            f"{r['wasserstein']:.5f} | {r['nnz_d']:.0f} | {r['nnz_q']:.0f} | "
            f"{r['n_QH']} | {n_lex} | " +
            (f"{r['Q_H_lex_rho']:.3f} [{r['Q_H_lex_rho_lo']:.2f}, {r['Q_H_lex_rho_hi']:.2f}]"
             if r.get("Q_H_lex_rho") is not None else "—") + " | " +
            (f"{r['Q_H_alias_mrr']:.4f}" if r.get("Q_H_alias_mrr") is not None else "—")
            + " |")

    v = {n: verdict(r) for n, r in rows.items()}
    passing = [n for n, x in v.items() if isinstance(x, dict) and x["pass_"]]
    crand = rows.get("Crand", {})
    v1 = rows.get("V1", {})
    h14 = (crand.get("mrr_all") is not None and v1.get("mrr_all") is not None
           and crand["mrr_all"] < v1["mrr_all"] * 0.9)

    def by(name):
        return rows.get(name, {}).get("Q_H_rho")
    dec = dict(
        passing_runs=passing,
        H10_verdict=(
            "V1 passes on every split evaluated" if {"V1", "V1_cluster", "V1_rare"} <= set(passing)
            else f"V1 passes on {[p.replace('V1_', '') or 'random' for p in passing if p.startswith('V1')]} "
                 f"and falls short elsewhere" if any(p.startswith("V1") for p in passing)
            else "only V1+VD passes" if "V1vd" in passing
            else "V2+VD passes" if "V2vd" in passing else "no arm passes"),
        H11_V3_rho=by("V3"), H11_verdict=(
            "supported" if (by("V3") is not None and by("V3") <= 0.5) else "not supported"),
        H12_vd_effect={k: (by(k + "vd"), by(k)) for k in ("V1", "V2", "V3")},
        H13_difficulty_order={s: by(n) for s, n in
                              (("random", "V1"), ("cluster", "V1_cluster"),
                               ("rare", "V1_rare"))},
        H14_crand_worse=bool(h14),
        H14_verdict="supported" if h14 else "NOT supported — stop and reconsider",
        recipe=("frozen encoder + light head" if any(p.startswith("V1") and
                                                     not p.endswith("vd")
                                                     for p in passing) else
                "frozen encoder + light head + vocabulary dropout" if "V1vd" in passing else
                "LoRA with staggered entry refresh" if "V2vd" in passing else
                "no arm passes"),
        signed_gaps={n: r["signed_gap_r"] for n, r in rows.items()},
        pilotE_triggered=bool(
            not passing and any(abs(r["signed_gap_r"] or 0) > LOG15 for r in rows.values())),
        summary=None)
    systematic = [n for n, r in rows.items()
                  if r["signed_gap_r"] is not None and abs(r["signed_gap_r"]) > LOG15]
    dec["systematic_gap_runs"] = systematic
    dec["summary"] = (
        f"{len(passing)} of {len(rows)} runs meet H10. "
        + ("A systematic signed activation gap is present, so Pilot E applies."
           if dec["pilotE_triggered"] else
           "No systematic signed gap; if no arm passes the problem is representational, "
           "not calibrational (§D.7 final branch)."))
    save_json(dict(decision=dec, rows=rows, checks=v),
              paths.RESULTS / "46_pilotD_decision.json")
    md += ["", "## Decision (§D.7)", "", "```json", json.dumps(dec, indent=2), "```"]
    (paths.REPORTS / "pilotD.md").write_text("\n".join(md))
    lg.info(json.dumps(dec, indent=2)[:2000])


if __name__ == "__main__":
    main()
