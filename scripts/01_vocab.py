"""§0.3 — build the vocabulary V and sample prototype occurrence sites.

Pass 1: count lowercase-alphabetic word occurrences over the prototype pool P.
        V = top 30k words with >= 100 occurrences in P.
Pass 2: bottom-k (random-key) reservoir of up to 100 occurrence pids per entry.
        Bottom-k with uniform random keys is an exact uniform sample without
        replacement and merges across workers by keeping the smallest keys.

Also emits the occurrence-count distribution per frequency decile, which §0.3
requires before V is frozen.
"""
from __future__ import annotations
import argparse, heapq, os, re, sys
from collections import Counter
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import Collection, load_splits
from dvlsr.util import get_logger, save_json, Timer, rng

lg = get_logger("vocab", "01_vocab.log")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
NPROC = 64

_COL = None
_P = None


def _init():
    global _COL, _P
    _COL = Collection()
    _P = load_splits()["P"]


def _count_shard(args):
    lo, hi = args
    c = Counter()
    col, P = _COL, _P
    for i in range(lo, hi):
        for w in WORD_RE.findall(col[int(P[i])]):
            if w.isalpha():
                c[w.lower()] += 1
    return c


def _sample_shard(args):
    """Bottom-k sample of occurrence pids per entry (max-heap of size K on random keys)."""
    lo, hi, vocab, K = args
    col, P = _COL, _P
    g = rng("proto-sample", lo)
    heaps: dict[int, list] = {}
    # a chunk of random keys drawn lazily
    buf, bi = g.random(1 << 16), 0
    for i in range(lo, hi):
        pid = int(P[i])
        seen = set()
        for w in WORD_RE.findall(col[pid]):
            if not w.isalpha():
                continue
            wl = w.lower()
            j = vocab.get(wl)
            if j is None or j in seen:
                continue
            seen.add(j)          # one sampling unit per (entry, passage)
            if bi >= len(buf):
                buf, bi = g.random(1 << 16), 0
            key = -float(buf[bi]); bi += 1        # negated: heapq min-heap -> max on |key|
            h = heaps.get(j)
            if h is None:
                heaps[j] = [(key, pid)]
            elif len(h) < K:
                heapq.heappush(h, (key, pid))
            elif key > h[0][0]:
                heapq.heapreplace(h, (key, pid))
    return heaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    vocab_npz = paths.ART / "vocab.npz"
    if vocab_npz.exists() and not a.force:
        lg.info("vocab exists; nothing to do")
        return

    P = load_splits()["P"]
    n = len(P)
    bounds = np.linspace(0, n, NPROC * 4 + 1).astype(int)
    chunks = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]

    with Timer("pass 1: count words over P", lg):
        with mp.get_context("fork").Pool(NPROC, initializer=_init) as pool:
            total = Counter()
            for c in pool.imap_unordered(_count_shard, chunks):
                total.update(c)
    lg.info(f"distinct alphabetic word types in P: {len(total):,}")

    # ---- freeze V ------------------------------------------------------------
    qualified = [(w, c) for w, c in total.items() if c >= paths.V_MIN_OCC]
    qualified.sort(key=lambda x: (-x[1], x[0]))
    lg.info(f"types with >= {paths.V_MIN_OCC} occurrences: {len(qualified):,}")
    V = qualified[: paths.V_SIZE]
    words = [w for w, _ in V]
    freq = np.asarray([c for _, c in V], dtype=np.int64)
    lg.info(f"|V| = {len(words)} (realized); freq range {freq[0]:,} .. {freq[-1]:,}")

    # occurrence-count distribution per frequency decile (the §0.3 pre-check)
    dec = np.minimum((np.arange(len(words)) * 10) // len(words), 9)
    decile_stats = []
    for d in range(10):
        f = freq[dec == d]
        decile_stats.append(dict(
            decile=d, n=int(len(f)), min=int(f.min()), p10=float(np.percentile(f, 10)),
            median=float(np.median(f)), p90=float(np.percentile(f, 90)), max=int(f.max())))
    for s in decile_stats:
        lg.info(f"decile {s['decile']}: n={s['n']} freq min={s['min']} med={s['median']:.0f} max={s['max']}")

    # ---- pass 2: prototype occurrence sites ---------------------------------
    vocab = {w: i for i, w in enumerate(words)}
    K = paths.PROTO_MAX_OCC
    args = [(lo, hi, vocab, K) for lo, hi in chunks]
    merged: dict[int, list] = {}
    with Timer("pass 2: sample occurrence sites", lg):
        with mp.get_context("fork").Pool(NPROC, initializer=_init) as pool:
            for heaps in pool.imap_unordered(_sample_shard, args):
                for j, h in heaps.items():
                    m = merged.get(j)
                    if m is None:
                        merged[j] = h
                    else:
                        m.extend(h)
                        if len(m) > 4 * K:
                            merged[j] = heapq.nlargest(K, m)
    occ_pid = np.full((len(words), K), -1, dtype=np.int32)
    occ_n = np.zeros(len(words), dtype=np.int32)
    for j, h in merged.items():
        top = heapq.nlargest(K, h)                       # K largest keys == bottom-k sample
        pids = np.asarray([p for _, p in top], dtype=np.int32)
        occ_pid[j, : len(pids)] = pids
        occ_n[j] = len(pids)
    lg.info(f"entries with < {K} sampled occurrences: {(occ_n < K).sum()} "
            f"(min {occ_n.min()})")

    # ---- stopword flag (§0.3): ~150-word list --------------------------------
    stop = set(STOPWORDS)
    is_stop = np.asarray([w in stop for w in words], dtype=bool)
    lg.info(f"stopwords present in V: {int(is_stop.sum())} / {len(stop)}")

    np.savez(vocab_npz, words=np.asarray(words), freq=freq, decile=dec.astype(np.int8),
             is_stop=is_stop, occ_pid=occ_pid, occ_n=occ_n)
    save_json(dict(size=len(words), min_occ=paths.V_MIN_OCC,
                   n_qualified=len(qualified), n_types=len(total),
                   freq_max=int(freq[0]), freq_min=int(freq[-1]),
                   n_stopwords=int(is_stop.sum()), decile_stats=decile_stats,
                   occ_min=int(occ_n.min()), occ_below_K=int((occ_n < K).sum())),
              paths.RESULTS / "01_vocab.json")
    lg.info(f"saved {vocab_npz}")


# a ~150-word English stopword list (Lucene/NLTK style union, trimmed)
STOPWORDS = """a about above after again against all am an and any are aren as at be because been
before being below between both but by can cannot could couldn did didn do does doesn doing don down
during each few for from further had hadn has hasn have haven having he her here hers herself him
himself his how i if in into is isn it its itself just ll me more most mustn my myself no nor not now
o of off on once only or other our ours ourselves out over own re s same shan she should shouldn so
some such t than that the their theirs them themselves then there these they this those through to
too under until up ve very was wasn we were weren what when where which while who whom why will with
won would wouldn y you your yours yourself yourselves also may many much us one two get like well
even back go make know take see come think look want give use find tell ask work seem feel try leave
call""".split()

if __name__ == "__main__":
    main()
