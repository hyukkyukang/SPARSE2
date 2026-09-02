"""Encode many Pilot D checkpoints at once: one configuration per GPU, three
worker processes inside it. Sharding a single encode across all GPUs leaves the
others idle between configurations; this keeps every GPU busy."""
from __future__ import annotations
import argparse, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("parenc", "47_parallel_encode.log")
ENC = str(paths.REPO / "scripts" / "43_encode_eval.py")


def done(name, kind):
    b = paths.EMB / f"D_{name}_{kind}"
    return b.with_name(b.name + "_idx.npy").exists()


def main(a):
    names = [n for n in sorted(os.listdir(paths.CKPT))
             if (paths.CKPT / n / "head.pt").exists()
             and not n.startswith(("lamtune", "smoke", "flops", "reftest"))]
    if a.only:
        names = [n for n in names if n in set(a.only.split(","))]
    if a.skip:
        names = [n for n in names if n not in set(a.skip.split(","))]
    gpus = a.gpus.split(",")
    lg.info(f"{len(names)} configurations on {len(gpus)} GPUs: {names}")
    todo, running = list(names), []
    env0 = dict(os.environ, PYTHONPATH=str(paths.REPO), OPENBLAS_NUM_THREADS="16")
    t0 = time.time()
    while todo or running:
        while todo and len(running) < len(gpus):
            used = {r[1] for r in running}
            g = next(x for x in gpus if x not in used)
            n = todo.pop(0)
            cmd = "; ".join(
                f"python3 {ENC} --name {n} --kind {k} --n-shards {s} --per-gpu {p}"
                for k, s, p in (("q", 2, 2), ("s", 3, 3), ("d", 3, 3))
                if not done(n, k))
            if not cmd:
                lg.info(f"[skip] {n} already encoded")
                continue
            log = open(paths.LOGS / f"parenc_{n}.log", "w")
            env = dict(env0, CUDA_VISIBLE_DEVICES=g)
            lg.info(f"[start] {n} on gpu {g}")
            running.append((subprocess.Popen(["bash", "-c", cmd], env=env,
                                             stdout=log, stderr=subprocess.STDOUT),
                            g, n, log))
        time.sleep(3)
        for r in list(running):
            p, g, n, log = r
            if p.poll() is not None:
                log.close(); running.remove(r)
                lg.info(f"[{'done' if p.returncode == 0 else 'FAILED'}] {n} "
                        f"(gpu {g}, {time.time()-t0:.0f}s elapsed)")
    lg.info(f"all encodes finished in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    main(ap.parse_args())
