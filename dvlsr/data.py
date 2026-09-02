"""MS MARCO v1 passage collection access and the P / Q / S splits of §0.2."""
from __future__ import annotations
import numpy as np
from pathlib import Path

from . import paths
from .util import get_logger, rng, save_json, load_json

lg = get_logger("data")

BIN = paths.PREP / "passages.bin"
OFF = paths.PREP / "offsets.npy"


def build_collection_binary(force: bool = False):
    """Concatenate the collection into one utf-8 blob + int64 offsets (mmap-friendly)."""
    if BIN.exists() and OFF.exists() and not force:
        return
    offs = [0]
    with open(paths.COLLECTION_TSV, "rb") as fi, open(BIN, "wb") as fo:
        pos = 0
        for i, line in enumerate(fi):
            pid, _, text = line.partition(b"\t")
            assert int(pid) == i, f"non-contiguous pid at line {i}: {pid!r}"
            text = text.rstrip(b"\n")
            fo.write(text)
            pos += len(text)
            offs.append(pos)
    np.save(OFF, np.asarray(offs, dtype=np.int64))
    lg.info(f"collection binary: {len(offs)-1} passages, {pos/1e9:.2f} GB")


class Collection:
    """Read-only mmap view of the passage collection; pid == row index."""

    def __init__(self):
        self.off = np.load(OFF, mmap_mode="r")
        self._buf = np.memmap(BIN, dtype=np.uint8, mode="r")
        self.n = len(self.off) - 1

    def __len__(self):
        return self.n

    def __getitem__(self, i: int) -> str:
        a, b = self.off[i], self.off[i + 1]
        return self._buf[a:b].tobytes().decode("utf-8", "replace")

    def texts(self, idx) -> list[str]:
        return [self[int(i)] for i in idx]


# --------------------------------------------------------------------------- splits
SPLITS = paths.PREP / "splits.npz"


def build_splits(force: bool = False):
    """Q (40k probe/bank pool), S (100k stats), P (prototype pool = all \\ Q \\ dev qrels).

    P/Q disjointness is what stops a probe's own passage from feeding its own entry (§0.2).
    S may overlap P: it is only ever used for corpus-level counts.
    """
    if SPLITS.exists() and not force:
        return dict(np.load(SPLITS))
    n = paths.N_PASSAGES
    g = rng("splits")
    perm = g.permutation(n)
    Q = np.sort(perm[: paths.N_Q]).astype(np.int32)
    S = np.sort(g.choice(n, size=paths.N_S, replace=False)).astype(np.int32)
    qrels_dev = np.unique(load_qrels(paths.QRELS_DEV_SMALL)[1]).astype(np.int32)
    excl = np.zeros(n, dtype=bool)
    excl[Q] = True
    excl[qrels_dev] = True
    P = np.flatnonzero(~excl).astype(np.int32)
    out = dict(P=P, Q=Q, S=S, qrels_dev_pids=qrels_dev)
    np.savez(SPLITS, **out)
    save_json({k: int(len(v)) for k, v in out.items()}, paths.RESULTS / "00_splits.json")
    lg.info({k: len(v) for k, v in out.items()})
    return out


def load_splits():
    return dict(np.load(SPLITS))


# --------------------------------------------------------------------------- queries / qrels
def load_queries(path: Path) -> tuple[np.ndarray, list[str]]:
    qids, texts = [], []
    with open(path) as f:
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            qids.append(int(a)); texts.append(b)
    return np.asarray(qids, dtype=np.int64), texts


def load_qrels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    q, d = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            q.append(int(p[0])); d.append(int(p[2]))
    return np.asarray(q, dtype=np.int64), np.asarray(d, dtype=np.int64)


def qrels_dict(path: Path) -> dict[str, dict[str, int]]:
    qs, ds = load_qrels(path)
    out: dict[str, dict[str, int]] = {}
    for a, b in zip(qs, ds):
        out.setdefault(str(a), {})[str(b)] = 1
    return out
