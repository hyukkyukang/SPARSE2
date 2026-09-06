"""Orchestrate Pilot D: lambda tuning, the §D.5 run list (plus the follow-up arms),
encoding and evaluation. Every stage is resumable: finished checkpoints, encodes and
evaluations are skipped unless --force."""
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
    tau = kw.pop("tau", cfg["tau"])          # R1 arms carry their own operating point
    rep = kw.pop("rep", cfg["rep"])
    a = [PY, TRAIN, "--name", name, "--encoder", cfg["encoder"],
         "--layer", str(cfg["layer"]), "--rep", rep,
         "--r1-prefix", cfg["r1_prefix"], "--tau", str(tau),
         "--lam-d", str(kw.pop("lam_d", cfg["lam_d"])),
         "--lam-q", str(kw.pop("lam_q", cfg["lam_q"]))]
    for k, v in kw.items():
        if k.startswith("_"):
            continue
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
    """§D.5 run list plus the follow-up arms. Entries are (name, train kwargs, oracle).

    rho compares a model to an oracle of its own architecture; V2 has no oracle in
    the protocol's run list, so it is paired with V1oracle and flagged (arch*) in the
    report. The distillation arm gets its own oracle because its training objective
    differs, and the denominator of rho is the oracle's own held-out benefit.
    """
    R = []

    def add(name, orc, **kw):            # kw may itself contain oracle=True (the training flag)
        R.append((name, kw, orc))

    S = "random"
    add("V1", "V1oracle", variant="V1", split=S)
    add("V1vd", "V1oracle", variant="V1", split=S, vd=True)
    # 32-query arms get the 2,000-step warmup of notes/deviations.md D11
    add("V2", "V1oracle", variant="V2", split=S, warmup=500, max_steps=3000)
    add("V2vd", "V1oracle", variant="V2", split=S, vd=True, warmup=500, max_steps=3000)
    add("V3", "V3oracle", variant="V3", split=S, warmup=500, max_steps=3000)
    add("V3vd", "V3oracle", variant="V3", split=S, vd=True, warmup=500, max_steps=3000)
    add("V1oracle", None, variant="V1", split=S, oracle=True)
    add("V3oracle", None, variant="V3", split=S, oracle=True, warmup=500, max_steps=3000)
    add("Crand", "V1oracle", variant="V1", split=S, crand=True)
    for s in ("cluster", "rare"):
        add(f"V1_{s}", f"V1oracle_{s}", variant="V1", split=s)
        add(f"V1vd_{s}", f"V1oracle_{s}", variant="V1", split=s, vd=True)
        add(f"V1oracle_{s}", None, variant="V1", split=s, oracle=True)
    add("V3_cluster", "V1oracle_cluster", variant="V3", split="cluster", warmup=500, max_steps=3000)

    # ---- follow-up arms (2026-09): semantic dropout, capacity control, distillation ----
    for s in (S, "cluster"):
        sfx = "" if s == S else f"_{s}"
        orc = "V1oracle" + sfx
        # cluster-wise vocabulary dropout: whole regions of the entry space per step
        add(f"V1cvd{sfx}", orc, variant="V1", split=s, vd=True, vd_mode="cluster")
        # cluster-VD + meta-held-out clusters with the calibration loss
        add(f"V1meta{sfx}", orc, variant="V1", split=s, vd=True, vd_mode="cluster",
            meta_frac=0.1)
        # linear head: can the over-firing be blamed on head capacity?
        add(f"V1lin{sfx}", orc, variant="V1", split=s, head="linear")
    # distillation from the encoder's own dense score (effectiveness ceiling of V1)
    add("V1dist", "V1dist_oracle", variant="V1", split=S, teacher="dense")
    add("V1dist_oracle", None, variant="V1", split=S, oracle=True, teacher="dense")
    # phrase / entity insertion pilot (extended vocabulary; V1_phrase is derived from
    # V1oracle by scripts/82_phrase_ckpt.py, not trained)
    add("V1oracle_phrase", None, variant="V1", split="phrase", oracle=True, vocab_tag="phr")
    add("V1_phrase", "V1oracle_phrase", variant="V1", split="phrase", vocab_tag="phr",
        _derived=True)

    # bare-string (R1) entries, trained: at layer 12 they beat prototypes *untrained*
    # (results_gpu10/31_pilotC_e5_L12_R1.json), which reopens one-forward-pass insertion
    r1 = paths.RESULTS / f"31_pilotC_{cfg['encoder']}_L{cfg['layer']}_R1.json"
    if r1.exists():
        t1 = json.load(open(r1))["tau_d"][str(cfg["nnz_d"])]["tau"]
        add("V1oracle_R1", None, variant="V1", split=S, oracle=True, rep="R1", tau=t1)
        add("V1_R1", "V1oracle_R1", variant="V1", split=S, rep="R1", tau=t1)
        add("V1oracle_cluster_R1", None, variant="V1", split="cluster", oracle=True, rep="R1", tau=t1)
        add("V1_cluster_R1", "V1oracle_cluster_R1", variant="V1", split="cluster", rep="R1", tau=t1)
        add("V1cvd_cluster_R1", "V1oracle_cluster_R1", variant="V1", split="cluster", rep="R1",
            tau=t1, vd=True, vd_mode="cluster")

    # entry side whitened with the token-side transform (independent of the seen set):
    # does seen-only entry whitening cause the cluster-split over-firing?
    add("V1oracle_Wh", None, variant="V1", split=S, oracle=True, entry_tf="shared")
    add("V1_Wh", "V1oracle_Wh", variant="V1", split=S, entry_tf="shared")
    add("V1oracle_cluster_Wh", None, variant="V1", split="cluster", oracle=True, entry_tf="shared")
    add("V1_cluster_Wh", "V1oracle_cluster_Wh", variant="V1", split="cluster", entry_tf="shared")

    # ---- Pilot F: per-entry background normalisation inside the score (notes/pilotF.md) ----
    # The cluster pair is the decisive test; the random pair says what the normalisation
    # costs where calibration was never the problem; the +Wh pair combines it with the
    # entry transform that Pilot D found is worth 23% MRR on its own.
    add("V1oracle_norm_cluster", None, variant="V1", split="cluster", oracle=True, entry_norm="q")
    add("V1norm_cluster", "V1oracle_norm_cluster", variant="V1", split="cluster", entry_norm="q")
    add("V1oracle_norm", None, variant="V1", split=S, oracle=True, entry_norm="q")
    add("V1norm", "V1oracle_norm", variant="V1", split=S, entry_norm="q")
    add("V1oracle_normWh_cluster", None, variant="V1", split="cluster", oracle=True,
        entry_norm="q", entry_tf="shared")
    add("V1normWh_cluster", "V1oracle_normWh_cluster", variant="V1", split="cluster",
        entry_norm="q", entry_tf="shared")
    # the moment-matching contrast the diagnostic predicts should not work
    add("V1oracle_normz_cluster", None, variant="V1", split="cluster", oracle=True, entry_norm="z")
    add("V1normz_cluster", "V1oracle_normz_cluster", variant="V1", split="cluster", entry_norm="z")

    # ---- Pilot G: what the Pilot F result implies is worth testing next ----
    # (a) bare-string entries fail on the cluster split *by over-firing* (gap_r +0.77,
    #     rho -0.05) and they retrieve better than prototypes; tail normalisation is
    #     aimed at exactly that failure, so the pair could give one-forward-pass
    #     insertion with usable calibration.
    if r1.exists():
        add("V1oracle_norm_cluster_R1", None, variant="V1", split="cluster", oracle=True,
            rep="R1", tau=t1, entry_norm="q")
        add("V1norm_cluster_R1", "V1oracle_norm_cluster_R1", variant="V1", split="cluster",
            rep="R1", tau=t1, entry_norm="q")
    # (b) the rare split is the motivating use case and the only one where recovery was
    #     already high; normalisation must not damage it.
    add("V1oracle_norm_rare", None, variant="V1", split="rare", oracle=True, entry_norm="q")
    add("V1norm_rare", "V1oracle_norm_rare", variant="V1", split="rare", entry_norm="q")
    # (c) the cost of a text-defined vocabulary, isolated: identical head, data and
    #     schedule, with the seen entry rows trained as free parameters.
    add("V1oracle_param", None, variant="V1", split=S, oracle=True, entry_param=True)
    add("V1param", "V1oracle_param", variant="V1", split=S, entry_param=True)

    # ---- Pilot H: seed evidence for the arms the *current* decision rests on ----
    # §D.2 requires two seeds for every arm a decision depends on. The decision moved from
    # V1-on-random (which has two) to the cluster pair with and without tail normalisation
    # (which have one each). The oracle is reused, so this is the model's seed variance.
    add("V1_cluster_s2", "V1oracle_cluster", variant="V1", split="cluster", seed=2)
    add("V1norm_cluster_s2", "V1oracle_norm_cluster", variant="V1", split="cluster",
        entry_norm="q", seed=2)

    # ---- Pilot I: is the gap to a learned vocabulary systematic or per-entry? ----
    # V1param (trained rows) buys +33% MRR@10 with L2-normalised rows, so the benefit is
    # entirely in the entry *directions*. A shared residual map on the entry side can
    # capture the systematic part of that and still apply to a newly inserted entry;
    # it cannot memorise per-entry idiosyncrasy. The pair of numbers decomposes the gap.
    add("V1oracle_eh", None, variant="V1", split=S, oracle=True, entry_head=True)
    add("V1eh", "V1oracle_eh", variant="V1", split=S, entry_head=True)
    add("V1oracle_eh_cluster", None, variant="V1", split="cluster", oracle=True, entry_head=True)
    add("V1eh_cluster", "V1oracle_eh_cluster", variant="V1", split="cluster", entry_head=True)
    # the entry map together with the correction that fixed the cluster split
    add("V1oracle_ehnorm_cluster", None, variant="V1", split="cluster", oracle=True,
        entry_head=True, entry_norm="q")
    add("V1ehnorm_cluster", "V1oracle_ehnorm_cluster", variant="V1", split="cluster",
        entry_head=True, entry_norm="q")

    # ---- Pilot J: is the residual recovery gap prototype noise? ----
    # k=100 prototypes from the second disjoint occurrence half (scripts/83_proto_k100.py),
    # no new encoding.
    add("V1oracle_k100_cluster", None, variant="V1", split="cluster", oracle=True,
        vocab_tag="k100")
    add("V1k100_cluster", "V1oracle_k100_cluster", variant="V1", split="cluster",
        vocab_tag="k100")

    for n in seeds_extra:
        kw, orc = next((k, o) for nm, k, o in R if nm == n)
        add(f"{n}_s2", orc, **dict(kw, seed=2))
    return R


def _ckpt(n):
    return (paths.CKPT / n / "head.pt").exists()


def _running(n):
    """A trainer that is still alive (its RUNNING marker holds a live PID)."""
    p = paths.CKPT / n / "RUNNING"
    if not p.exists():
        return False
    try:
        os.kill(int(p.read_text().strip()), 0)
        return True
    except Exception:
        return False


def _enc_marker(n, kind):
    return paths.EMB / f"D_{n}_{kind}.done"


def main(a):
    cfg = json.load(open(paths.RESULTS / "32_pilotC_decision.json"))
    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else None
    only = set(a.only.split(",")) if a.only else None
    R = run_list(cfg)
    if a.list:
        for n, kw, o in R:
            print(f"{n:22s} oracle={o!s:18s} ckpt={_ckpt(n)!s:5s} "
                  f"enc={all(_enc_marker(n, k).exists() for k in 'qds')!s:5s} "
                  f"eval={(paths.RESULTS / f'44_pilotD_{n}.json').exists()!s:5s} {kw}")
        return
    if a.stage in ("all", "lambda"):
        ld, lq = lambda_sweep(cfg, gpus)
        cfg["lam_d"], cfg["lam_q"] = ld, lq
        save_json(cfg, paths.RESULTS / "32_pilotC_decision.json")
    if a.stage in ("all", "train"):
        jobs = []
        for n, kw, _ in R:
            if only and n not in only:
                continue
            if kw.get("_derived"):
                continue
            if _ckpt(n) and not a.force:
                lg.info(f"[train] {n}: checkpoint exists, skipping")
                continue
            if _running(n):
                lg.info(f"[train] {n}: already running elsewhere, skipping")
                continue
            extra = []
            if a.batch_queries and kw.get("variant", "V1") == "V1":
                extra += ["--batch-queries", str(a.batch_queries)]
            if a.enc_chunk:
                extra += ["--enc-chunk", str(a.enc_chunk)]
            jobs.append((n, base_args(cfg, n, **kw) + extra))
        lg.info(f"[train] {len(jobs)} runs on {gpus or 'all visible'} gpus, "
                f"{a.train_per_gpu} per gpu: {[j[0] for j in jobs]}")
        with Timer("Pilot D training runs", lg):
            done, failed = run_queue(jobs, gpus=gpus, per_gpu=a.train_per_gpu, logger=lg)
        save_json(dict(done=done, failed=failed), paths.RESULTS / "45_train_status.json")
        if failed:
            lg.error(f"[train] failed: {failed}")
            sys.exit(1)
    if a.stage in ("all", "encode"):
        # one configuration at a time, but every GPU on it: sharding one encode over
        # all workers is much faster than running several encodes one-per-GPU.
        import subprocess
        names = [n for n, _, _ in R if (not only or n in only)]
        missing = [n for n in names if not _ckpt(n)]
        if missing:
            lg.warning(f"skipping (no checkpoint): {missing}")
        names = [n for n in names if n not in set(missing)]
        done, failed = [], []
        env = dict(os.environ, PYTHONPATH=str(paths.REPO))
        for n in names:
            for kind, ns in (("q", 8), ("d", a.enc_shards), ("s", a.enc_shards)):
                tag = f"enc_{n}_{kind}"
                if _enc_marker(n, kind).exists() and not a.force:
                    continue
                with open(paths.LOGS / f"{tag}.log", "w") as log:
                    r = subprocess.run([PY, ENC, "--name", n, "--kind", kind,
                                        "--n-shards", str(ns), "--per-gpu", str(a.enc_per_gpu)],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
                if r.returncode == 0:
                    _enc_marker(n, kind).touch()
                    done.append(tag)
                else:
                    failed.append(tag)
                lg.info(f"[encode] {tag} rc={r.returncode}")
        save_json(dict(done=done, failed=failed), paths.RESULTS / "45_encode_status.json")
        if failed:
            lg.error(f"[encode] failed: {failed}")
            sys.exit(1)
    if a.stage in ("all", "eval"):
        pairs = []
        for n, kw, orc in R:
            if orc is None or (only and n not in only):
                continue
            ready = (_ckpt(n) and _ckpt(orc)
                     and all(_enc_marker(x, k).exists() for x in (n, orc) for k in "qds"))
            if not ready:
                lg.info(f"[eval] {n}: model or oracle {orc} not encoded yet, skipping")
                continue
            if (paths.RESULTS / f"44_pilotD_{n}.json").exists() and not a.force:
                lg.info(f"[eval] {n}: result exists, skipping")
                continue
            pairs.append((n, orc, kw["split"]))
        jobs = [(f"eval_{n}", [PY, EVAL, "--name", n, "--oracle", o, "--split", s])
                for n, o, s in pairs]
        with Timer("Pilot D evaluation", lg):
            done, failed = run_queue(jobs, gpus=gpus, logger=lg)
        save_json(dict(done=done, failed=failed), paths.RESULTS / "45_eval_status.json")
        if failed:
            lg.error(f"[eval] failed: {failed}")
            sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "lambda", "train", "encode", "eval"])
    ap.add_argument("--gpus", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--enc-shards", type=int, default=24)
    ap.add_argument("--enc-per-gpu", type=int, default=3)
    ap.add_argument("--train-per-gpu", type=int, default=2,
                    help="concurrent training runs per GPU (a frozen-encoder run needs ~8 GB)")
    ap.add_argument("--batch-queries", type=int, default=0)
    ap.add_argument("--enc-chunk", type=int, default=0)
    main(ap.parse_args())
