"""Orchestrate Pilot D: lambda tuning, the §D.5 run list, encoding and evaluation."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json, run_queue, Timer

lg = get_logger("runD", "45_run_pilotD.log")
PY = sys.executable
TRAIN = str(paths.REPO / "scripts" / "42_train.py")
ENC = str(paths.REPO / "scripts" / "43_encode_eval.py")
EVAL = str(paths.REPO / "scripts" / "44_pilotD_eval.py")


def base_args(cfg, name, **kw):
    a = [PY, TRAIN, "--name", name, "--encoder", cfg["encoder"],
         "--layer", str(cfg["layer"]), "--rep", cfg["rep"],
         "--r1-prefix", cfg["r1_prefix"], "--tau", str(cfg["tau"]),
         "--lam-d", str(kw.pop("lam_d", cfg["lam_d"])),
         "--lam-q", str(kw.pop("lam_q", cfg["lam_q"]))]
    for k, v in kw.items():
        f = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                a.append(f)
        else:
            a += [f, str(v)]
    return a


def lambda_sweep(cfg, gpus, steps_frac=0.1):
    """§D.2: tune lambda on V1 only; keep the pair landing nearest C's nnz targets."""
    jobs = []
    for ld in (1e-4, 3e-4, 1e-3):
        n = f"lamtune_d{ld:g}"
        jobs.append((n, base_args(cfg, n, variant="V1", split="random",
                                  lam_d=ld, lam_q=3 * ld, epochs=1,
                                  seed=1) + ["--warmup", "200"]))
    lg.info(f"lambda sweep: {[j[0] for j in jobs]}")
    run_queue(jobs, gpus=gpus, logger=lg)
    best, bestd = None, 1e18
    out = {}
    for ld in (1e-4, 3e-4, 1e-3):
        n = f"lamtune_d{ld:g}"
        h = json.load(open(paths.CKPT / n / "history.json"))
        tail = h[-10:]
        nd = float(np.mean([x["nnz_d"] for x in tail]))
        nq = float(np.mean([x["nnz_q"] for x in tail]))
        d = abs(np.log((nd + 1) / paths.NNZ_DOC_TARGET)) + \
            abs(np.log((nq + 1) / paths.NNZ_QRY_TARGET))
        out[f"{ld:g}"] = dict(nnz_d=nd, nnz_q=nq, dist=d)
        lg.info(f"lambda_d={ld:g}: nnz_d={nd:.0f} nnz_q={nq:.0f} dist={d:.3f}")
        if d < bestd:
            best, bestd = ld, d
    save_json(dict(sweep=out, chosen_lam_d=best, chosen_lam_q=3 * best),
              paths.RESULTS / "45_lambda_sweep.json")
    return best, 3 * best


def run_list(cfg, seeds_extra=("V1", "V1vd")):
    """§D.5 run list."""
    R = []
    S = "random"
    R += [("V1", dict(variant="V1", split=S)),
          ("V1vd", dict(variant="V1", split=S, vd=True)),
          ("V2", dict(variant="V2", split=S)),
          ("V2vd", dict(variant="V2", split=S, vd=True)),
          ("V3", dict(variant="V3", split=S)),
          ("V3vd", dict(variant="V3", split=S, vd=True)),
          ("V1oracle", dict(variant="V1", split=S, oracle=True)),
          ("V3oracle", dict(variant="V3", split=S, oracle=True)),
          ("Crand", dict(variant="V1", split=S, crand=True))]
    for s in ("cluster", "rare"):
        R += [(f"V1_{s}", dict(variant="V1", split=s)),
              (f"V1vd_{s}", dict(variant="V1", split=s, vd=True)),
              (f"V1oracle_{s}", dict(variant="V1", split=s, oracle=True))]
    R += [("V3_cluster", dict(variant="V3", split="cluster"))]
    for n in seeds_extra:
        base = dict(R)[n]
        R.append((f"{n}_s2", dict(base, seed=2)))
    return R


def main(a):
    cfg = json.load(open(paths.RESULTS / "32_pilotC_decision.json"))
    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else None
    if a.stage in ("all", "lambda"):
        ld, lq = lambda_sweep(cfg, gpus)
        cfg["lam_d"], cfg["lam_q"] = ld, lq
        save_json(cfg, paths.RESULTS / "32_pilotC_decision.json")
    if a.stage in ("all", "train"):
        jobs = [(n, base_args(cfg, n, **kw)) for n, kw in run_list(cfg)]
        if a.only:
            keep = set(a.only.split(","))
            jobs = [j for j in jobs if j[0] in keep]
        with Timer("Pilot D training runs", lg):
            done, failed = run_queue(jobs, gpus=gpus, logger=lg)
        save_json(dict(done=done, failed=failed), paths.RESULTS / "45_train_status.json")
    if a.stage in ("all", "encode"):
        # one configuration at a time, but every GPU on it: sharding one encode over
        # 24 workers is ~8x faster than running 8 encodes one-per-GPU.
        import subprocess
        names = [n for n, _ in run_list(cfg)]
        if a.only:
            names = [n for n in names if n in set(a.only.split(","))]
        done, failed = [], []
        env = dict(os.environ, PYTHONPATH=str(paths.REPO))
        for n in names:
            for kind, ns in (("q", 8), ("d", 24), ("s", 24)):
                tag = f"enc_{n}_{kind}"
                with open(paths.LOGS / f"{tag}.log", "w") as log:
                    r = subprocess.run([PY, ENC, "--name", n, "--kind", kind,
                                        "--n-shards", str(ns), "--per-gpu", "3"],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
                (done if r.returncode == 0 else failed).append(tag)
                lg.info(f"[encode] {tag} rc={r.returncode}")
        save_json(dict(done=done, failed=failed), paths.RESULTS / "45_encode_status.json")
    if a.stage in ("all", "eval"):
        pairs = []
        for n, kw in run_list(cfg):
            if "oracle" in n:
                continue
            sp = kw["split"]
            orc = "V1oracle" if sp == "random" else f"V1oracle_{sp}"
            if kw["variant"] == "V3" and sp == "random":
                orc = "V3oracle"
            pairs.append((n, orc, sp))
        if a.only:
            keep = set(a.only.split(","))
            pairs = [p for p in pairs if p[0] in keep]
        jobs = [(f"eval_{n}", [PY, EVAL, "--name", n, "--oracle", o, "--split", s])
                for n, o, s in pairs]
        with Timer("Pilot D evaluation", lg):
            done, failed = run_queue(jobs, gpus=gpus, logger=lg)
        save_json(dict(done=done, failed=failed), paths.RESULTS / "45_eval_status.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "lambda", "train", "encode", "eval"])
    ap.add_argument("--gpus", default="")
    ap.add_argument("--only", default="")
    main(ap.parse_args())
