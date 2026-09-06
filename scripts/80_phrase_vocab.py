"""Phrase / entity insertion pilot, step 1: an extended vocabulary V_phr = V + phrases.

The protocol's goal statement is "arbitrary text units", but every pilot used single
lowercase words. Here the inserted entries are Title-case bigrams ("new york",
"harry potter", "windows vista") -- a cheap proxy for named entities and product
names, the terms the dynamic-vocabulary idea was motivated by.

Phrases are counted over the prototype pool P exactly as words were (§0.3): both
tokens Title-case, neither a stopword, >= 100 occurrences, top N_PHR by frequency.
Up to 100 occurrence passages per phrase are sampled (bottom-k with random keys).
Each phrase gets a frequency decile mapped onto the *word* vocabulary's decile
frequency ranges, so held-vs-seen comparisons stay within decile (§D.6.5).

Outputs
  artifacts/vocab_phr.npz          words+phrases, freq, decile, is_stop, occ_pid, occ_n, is_phrase
  prep/vocab_splits_phr.npz        'phrase' = the phrase rows; the §D.4 splits padded with False
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

lg = get_logger("phrvocab", "80_phrase_vocab.log")
TOK_RE = re.compile(r"[A-Za-z0-9]+")
TITLE_RE = re.compile(r"^[A-Z][a-z]{1,}$")
NPROC = int(os.environ.get("DVLSR_NPROC", "48"))
N_PHR = 2000

_COL = _P = _STOP = None


def _init(stop):
    global _COL, _P, _STOP
    _COL = Collection()
    _P = load_splits()["P"]
    _STOP = stop


def _pairs(text):
    toks = TOK_RE.findall(text)
    for a, b in zip(toks, toks[1:]):
        if TITLE_RE.match(a) and TITLE_RE.match(b):
            al, bl = a.lower(), b.lower()
            if al in _STOP or bl in _STOP:
                continue
            yield al + " " + bl


def _count_shard(args):
    lo, hi = args
    c = Counter()
    for i in range(lo, hi):
        for k in set(_pairs(_COL[int(_P[i])])):
            c[k] += 1
    return c


def _sample_shard(args):
    lo, hi, vocab, K = args
    g = rng("phrase-sample", lo)
    heaps: dict[int, list] = {}
    buf, bi = g.random(1 << 16), 0
    for i in range(lo, hi):
        pid = int(_P[i])
        for k in set(_pairs(_COL[pid])):
            j = vocab.get(k)
            if j is None:
                continue
            if bi >= len(buf):
                buf, bi = g.random(1 << 16), 0
            key = -float(buf[bi]); bi += 1
            h = heaps.get(j)
            if h is None:
                heaps[j] = [(key, pid)]
            elif len(h) < K:
                heapq.heappush(h, (key, pid))
            elif key > h[0][0]:
                heapq.heapreplace(h, (key, pid))
    return heaps


def main(force=False):
    out = paths.vocab_file("phr")
    if out.exists() and not force:
        lg.info("exists"); return
    z = np.load(paths.ART / "vocab.npz")
    words = [str(w) for w in z["words"]]
    freq, decile, is_stop = z["freq"], z["decile"], z["is_stop"]
    stop = {w for w, s in zip(words, is_stop) if s}
    P = load_splits()["P"]
    bounds = np.linspace(0, len(P), NPROC * 4 + 1).astype(int)
    chunks = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
    with Timer("count Title-case bigrams over P", lg):
        with mp.get_context("fork").Pool(NPROC, initializer=_init, initargs=(stop,)) as pool:
            total = Counter()
            for c in pool.imap_unordered(_count_shard, chunks):
                total.update(c)
    lg.info(f"distinct Title-case bigram types: {len(total):,}")
    wordset = set(words)
    qualified = [(k, c) for k, c in total.items() if c >= paths.V_MIN_OCC]
    qualified.sort(key=lambda x: (-x[1], x[0]))
    lg.info(f"bigrams with >= {paths.V_MIN_OCC} passages: {len(qualified):,}")
    phrases = [k for k, _ in qualified[:N_PHR]]
    pfreq = np.asarray([c for _, c in qualified[:N_PHR]], np.int64)
    lg.info(f"top phrases: {phrases[:25]}")

    # ---- occurrence sample ----
    vocab = {k: i for i, k in enumerate(phrases)}
    K = paths.PROTO_MAX_OCC
    merged: dict[int, list] = {}
    with Timer("sample occurrence passages per phrase", lg):
        with mp.get_context("fork").Pool(NPROC, initializer=_init, initargs=(stop,)) as pool:
            for heaps in pool.imap_unordered(_sample_shard,
                                             [(lo, hi, vocab, K) for lo, hi in chunks]):
                for j, h in heaps.items():
                    m = merged.get(j)
                    if m is None:
                        merged[j] = h
                    else:
                        m.extend(h)
                        if len(m) > 4 * K:
                            merged[j] = heapq.nlargest(K, m)
    occ_pid = np.full((len(phrases), K), -1, np.int32)
    occ_n = np.zeros(len(phrases), np.int32)
    for j, h in merged.items():
        top = heapq.nlargest(K, h)
        pids = np.asarray([p for _, p in top], np.int32)
        occ_pid[j, : len(pids)] = pids
        occ_n[j] = len(pids)

    # ---- frequency decile mapped onto the word vocabulary's decile ranges ----
    dec_min = np.asarray([freq[decile == d].min() for d in range(10)])
    pdec = np.full(len(phrases), 9, np.int8)
    for d in range(10):
        pdec[pfreq >= dec_min[d]] = np.minimum(pdec[pfreq >= dec_min[d]], d)
    # (deciles are ordered by decreasing frequency: decile 0 = most frequent)

    n_w, n_p = len(words), len(phrases)
    np.savez(out,
             words=np.asarray(words + phrases),
             freq=np.concatenate([freq, pfreq]),
             decile=np.concatenate([decile.astype(np.int8), pdec]),
             is_stop=np.concatenate([is_stop, np.zeros(n_p, bool)]),
             occ_pid=np.concatenate([z["occ_pid"], occ_pid]),
             occ_n=np.concatenate([z["occ_n"], occ_n]),
             is_phrase=np.concatenate([np.zeros(n_w, bool), np.ones(n_p, bool)]))
    sp = np.load(paths.vsplits_file(""))
    pad = np.zeros(n_p, bool)
    np.savez(paths.vsplits_file("phr"),
             phrase=np.concatenate([np.zeros(n_w, bool), np.ones(n_p, bool)]),
             random=np.concatenate([sp["random"], pad]),
             cluster=np.concatenate([sp["cluster"], pad]),
             rare=np.concatenate([sp["rare"], pad]),
             cluster_labels=np.concatenate([sp["cluster_labels"], np.full(n_p, -1)]))
    save_json(dict(n_phrases=n_p, n_bigram_types=len(total), n_qualified=len(qualified),
                   freq_max=int(pfreq[0]), freq_min=int(pfreq[-1]),
                   by_decile={int(d): int((pdec == d).sum()) for d in range(10)},
                   occ_min=int(occ_n.min()), examples=phrases[:60]),
              paths.RESULTS / "80_phrase_vocab.json")
    lg.info(f"saved {out}: {n_w} words + {n_p} phrases; phrase deciles "
            f"{ {int(d): int((pdec == d).sum()) for d in range(10)} }")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
