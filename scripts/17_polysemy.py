"""Build the §A.1 polysemy set: 30 words x 2 senses x 40 occurrences from P.

Labelling uses each sense's `indicators`, which are disjoint from the `anchors`
used for scoring (see notes/deviations.md D1). A passage is labelled for a sense
only if it contains an indicator of that sense and none of the other sense's.
"""
from __future__ import annotations
import argparse, json, os, re, sys
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import Collection, load_splits
from dvlsr.util import get_logger, rng, save_json, Timer

lg = get_logger("polysemy", "17_polysemy.log")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
N_PER_SENSE = 40
CAND_CAP = 400

_COL = _P = _CFG = None


def _init():
    global _COL, _P, _CFG
    _COL = Collection()
    _P = load_splits()["P"]
    _CFG = json.load(open(paths.REPO / "configs" / "polysemy.json"))["words"]


def _scan(args):
    lo, hi = args
    cfg = _CFG
    targets = set(cfg)
    ind = {w: {s: set(cfg[w][s]["indicators"]) for s in "AB"} for w in cfg}
    hits = {(w, s): [] for w in cfg for s in "AB"}
    for i in range(lo, hi):
        pid = int(_P[i])
        toks = {t.lower() for t in WORD_RE.findall(_COL[pid]) if t.isalpha()}
        for w in targets & toks:
            a = bool(ind[w]["A"] & toks)
            b = bool(ind[w]["B"] & toks)
            if a != b:
                s = "A" if a else "B"
                if len(hits[(w, s)]) < CAND_CAP:
                    hits[(w, s)].append(pid)
    return hits


def main(force=False):
    out = paths.ART / "polysemy.npz"
    if out.exists() and not force:
        lg.info("exists"); return
    cfg = json.load(open(paths.REPO / "configs" / "polysemy.json"))["words"]
    P = load_splits()["P"]
    n = len(P)
    NPROC = 64
    b = np.linspace(0, n, NPROC * 2 + 1).astype(int)
    chunks = [(int(b[i]), int(b[i + 1])) for i in range(len(b) - 1)]
    merged = {(w, s): [] for w in cfg for s in "AB"}
    with Timer("scan P for polysemous occurrences", lg):
        with mp.get_context("fork").Pool(NPROC, initializer=_init) as pool:
            for h in pool.imap_unordered(_scan, chunks):
                for k, v in h.items():
                    merged[k].extend(v)
    g = rng("polysemy")
    words, senses, pids = [], [], []
    thin = []
    for (w, s), v in sorted(merged.items()):
        v = np.unique(np.asarray(v, np.int32))
        if len(v) < N_PER_SENSE:
            thin.append((w, s, len(v)))
        take = v[g.choice(len(v), size=min(N_PER_SENSE, len(v)), replace=False)]
        for pid in take:
            words.append(w); senses.append(s); pids.append(int(pid))
    z = np.load(paths.ART / "vocab.npz")
    V = {str(x): i for i, x in enumerate(z["words"])}
    anchors = {}
    for w in cfg:
        for s in "AB":
            a = [x for x in cfg[w][s]["anchors"] if x in V][:5]
            assert len(a) == 5, (w, s, a)
            anchors[f"{w}|{s}"] = [V[x] for x in a]
    np.savez(out, words=np.asarray(words), senses=np.asarray(senses),
             pids=np.asarray(pids, np.int32),
             anchor_keys=np.asarray(list(anchors)),
             anchor_ids=np.asarray(list(anchors.values()), np.int32))
    save_json(dict(n_occ=len(pids), n_words=len(cfg),
                   thin=[list(t) for t in thin],
                   per_pair={f"{w}|{s}": int(len(np.unique(v))) for (w, s), v in merged.items()}),
              paths.RESULTS / "17_polysemy.json")
    lg.info(f"polysemy occurrences: {len(pids)}; thin pairs: {thin}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
