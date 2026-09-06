"""Run Pilot E's calibration variants end to end and apply the §E.3 rule."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("runE", "51_run_pilotE.log")
PY = sys.executable
ENC = str(paths.REPO / "scripts" / "43_encode_eval.py")
EVAL = str(paths.REPO / "scripts" / "44_pilotD_eval.py")
CAL = str(paths.REPO / "scripts" / "50_pilotE.py")
VARIANTS = ["raw", "Z", "DF", "SA"]


def main(a):
    env = dict(os.environ, PYTHONPATH=str(paths.REPO))
    cfg = json.load(open(paths.CKPT / a.name / "config.json"))
    split = cfg["split"]
    oracle = a.oracle or ("V1oracle" if split == "random" else f"V1oracle_{split}")
    ns_d, pg = a.enc_shards, a.enc_per_gpu
    if not (paths.ART / "calib" / f"{a.name}_Z.npy").exists():
        with Timer("compute calibration vectors", lg):
            subprocess.run([PY, CAL, "--name", a.name], env=env, check=True)
    res = {}
    for v in VARIANTS:
        tag = f"{a.name}_E{v}"
        calib = str(paths.ART / "calib" / f"{a.name}_{v}.npy")
        for kind, ns in (("q", 8), ("d", ns_d), ("s", ns_d)):
            with open(paths.LOGS / f"encE_{tag}_{kind}.log", "w") as log:
                subprocess.run([PY, ENC, "--name", a.name, "--kind", kind,
                                "--n-shards", str(ns), "--per-gpu", str(pg),
                                "--calib", calib, "--out-tag", tag],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with open(paths.LOGS / f"evalE_{tag}.log", "w") as log:
            subprocess.run([PY, EVAL, "--name", tag, "--oracle", oracle,
                            "--split", split], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        r = json.load(open(paths.RESULTS / f"44_pilotD_{tag}.json"))
        res[v] = r
        lg.info(f"[E:{v}] rho={r.get('Q_H', {}).get('rho', {}).get('rho')} "
                f"gap_r={r['activation']['signed_gap_r']:.4f} nnz_d={r['nnz_d']:.0f}")

    base = res["raw"]
    rows = {}
    for v, r in res.items():
        qh = r.get("Q_H", {})
        rows[v] = dict(rho=qh.get("rho", {}).get("rho"),
                       rho_lo=qh.get("rho", {}).get("lo"),
                       rho_hi=qh.get("rho", {}).get("hi"),
                       gap_to_oracle=qh.get("gap_to_oracle_rel"),
                       signed_gap_r=r["activation"]["signed_gap_r"],
                       gap_w=r["activation"]["gap_w"],
                       nnz_d=r["nnz_d"],
                       delta_nnz_d=r["nnz_d"] - base["nnz_d"])
    ok = [v for v in VARIANTS if (rows[v]["rho"] or -9) >= 0.8]
    dec = dict(rows=rows,
               chosen_variant=(ok[0] if ok else None),
               H15_verdict=("Z closes the gap" if rows["Z"]["rho"] and rows["raw"]["rho"]
                            and rows["Z"]["rho"] - rows["raw"]["rho"] >=
                            0.8 * (0.8 - rows["raw"]["rho"]) else "not supported"),
               verdict=("adopt " + ok[0] if ok else
                        "no variant reaches 0.8; " +
                        ("some reach 0.6, calibration helps but does not close the gap"
                         if any((rows[v]["rho"] or -9) >= 0.6 for v in VARIANTS)
                         else "the gap is representational (§D.7 final branch)")))
    save_json(dec, paths.RESULTS / "51_pilotE_decision.json")
    md = ["# Pilot E — insertion-time calibration", "",
          "| variant | rho | 95% CI | gap to oracle | signed gap_r | gap_w | nnz(d) | Δnnz(d) |",
          "|" + "---|" * 8]
    for v in VARIANTS:
        r = rows[v]
        md.append(f"| {v} | " + (f"{r['rho']:.3f}" if r["rho"] is not None else "—")
                  + " | " + (f"[{r['rho_lo']:.2f}, {r['rho_hi']:.2f}]"
                             if r["rho_lo"] is not None else "—")
                  + " | " + (f"{r['gap_to_oracle']:.3f}"
                             if r["gap_to_oracle"] is not None else "—")
                  + f" | {r['signed_gap_r']:.4f} | {r['gap_w']:.4f} | "
                  f"{r['nnz_d']:.0f} | {r['delta_nnz_d']:+.0f} |")
    md += ["", "A variant that 'fixes' rho by making documents denser has not fixed",
           "anything — read Δnnz(d) beside every rho.", "",
           "## Decision (§E.3)", "", "```json", json.dumps(dec, indent=2), "```"]
    (paths.REPORTS / "pilotE.md").write_text("\n".join(md))
    lg.info(json.dumps(dec["verdict"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--oracle", default="")
    ap.add_argument("--enc-shards", type=int, default=24)
    ap.add_argument("--enc-per-gpu", type=int, default=3)
    main(ap.parse_args())
