"""Logging, JSON bookkeeping, seeds, and a small multi-GPU shard runner."""
from __future__ import annotations
import json, logging, os, random, sys, time, hashlib, subprocess
from pathlib import Path
import numpy as np

from . import paths


def get_logger(name: str = "dvlsr", logfile: str | None = None) -> logging.Logger:
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname).1s %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    if logfile:
        fh = logging.FileHandler(paths.LOGS / logfile)
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    return lg


def set_seed(seed: int = paths.SEED):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def rng(*keys) -> np.random.Generator:
    """Deterministic generator keyed by strings/ints, so every artifact is reproducible."""
    h = hashlib.sha256(("|".join(map(str, keys)) + f"|{paths.SEED}").encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def save_json(obj, path: Path | str, **kw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default, **kw)
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def load_json(path: Path | str):
    with open(path) as f:
        return json.load(f)


class Timer:
    def __init__(self, msg, logger=None):
        self.msg, self.lg = msg, logger or get_logger()

    def __enter__(self):
        self.t = time.time(); self.lg.info(f"[start] {self.msg}"); return self

    def __exit__(self, *a):
        self.lg.info(f"[done ] {self.msg} ({time.time()-self.t:.1f}s)")


def gpu_map(worker_path: str, n_shards: int, extra_args: list[str] | None = None,
            gpus: list[int] | None = None, logger=None, per_gpu: int = 1):
    """Run `python worker_path --shard i --n-shards N` once per shard, pinned to a GPU.

    Simple process-level parallelism: robust (no CUDA-in-fork issues) and lets each
    worker own a whole GPU.
    """
    lg = logger or get_logger()
    if gpus is None:
        import torch
        gpus = list(range(torch.cuda.device_count()))
    slots = [g for g in gpus for _ in range(per_gpu)]
    procs, running = [], []
    todo = list(range(n_shards))
    env_base = dict(os.environ)
    env_base["PYTHONPATH"] = str(paths.REPO) + ":" + env_base.get("PYTHONPATH", "")
    while todo or running:
        while todo and len(running) < len(slots):
            used = [r[1] for r in running]
            free = next(g for g in slots if used.count(g) < per_gpu)
            s = todo.pop(0)
            env = dict(env_base); env["CUDA_VISIBLE_DEVICES"] = str(free)
            cmd = [sys.executable, worker_path, "--shard", str(s), "--n-shards", str(n_shards)]
            if extra_args:
                cmd += extra_args
            lg.info(f"launch shard {s} on gpu {free}: {' '.join(cmd[1:])}")
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            running.append((p, free, s))
        time.sleep(1.0)
        for r in list(running):
            p, g, s = r
            if p.poll() is not None:
                out = p.stdout.read()
                if p.returncode != 0:
                    lg.error(f"shard {s} FAILED (gpu {g}):\n{out[-4000:]}")
                    for pp, _, _ in running:
                        pp.kill()
                    raise RuntimeError(f"shard {s} failed")
                lg.info(f"shard {s} ok (gpu {g}) {out.strip()[-300:]}")
                running.remove(r)
                procs.append(s)
    return procs
