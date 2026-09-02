"""Probe set, background token bank B_H, and query-side bank B_Q (§0.2, §A.1).

Pass 1 (CPU, regex): choose which (pid, entry) occurrences will be probes and
which passages feed the bank -- all from Q only.
Pass 2 (GPU): encode those texts once and keep word-unit states at layers 4..12.
"""
from __future__ import annotations
import argparse, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import Collection, load_splits, load_queries
from dvlsr.encoders import Encoder
from dvlsr.util import get_logger, rng, save_json, Timer

lg = get_logger("probes", "10_probes_banks.log")
WORD_RE = re.compile(r"[A-Za-z0-9]+")

N_PROBE_CONTENT_PER_DECILE = 4000
N_PROBE_STOP = 10_000
N_TUNE = 5000
BANK_PASSAGES = 12_000          # ~30 units each -> >200k units


def plan(force=False):
    """Pass 1: pick probe occurrences and bank passages (encoder-independent)."""
    p = paths.ART / "probe_plan.npz"
    if p.exists() and not force:
        return dict(np.load(p, allow_pickle=True))
    z = np.load(paths.ART / "vocab.npz")
    words = [str(w) for w in z["words"]]
    vocab = {w: i for i, w in enumerate(words)}
    decile, is_stop = z["decile"], z["is_stop"]
    col = Collection()
    Q = load_splits()["Q"]

    # every (pid, entry) occurrence available in Q
    per_dec = [[] for _ in range(10)]
    stop_occ = []
    for pid in Q:
        seen = set()
        for w in WORD_RE.findall(col[int(pid)]):
            if not w.isalpha():
                continue
            j = vocab.get(w.lower())
            if j is None or j in seen:
                continue
            seen.add(j)
            (stop_occ if is_stop[j] else per_dec[decile[j]]).append((int(pid), j))
    g = rng("probe-plan")
    picks = []
    stats = {}
    for d in range(10):
        arr = np.asarray(per_dec[d], dtype=np.int64)
        n = min(N_PROBE_CONTENT_PER_DECILE, len(arr))
        sel = g.choice(len(arr), size=n, replace=False)
        picks.append(arr[sel])
        stats[f"decile{d}_available"] = len(arr)
        stats[f"decile{d}_used"] = int(n)
    arr = np.asarray(stop_occ, dtype=np.int64)
    n = min(N_PROBE_STOP, len(arr))
    sel = g.choice(len(arr), size=n, replace=False)
    picks.append(arr[sel])
    stats["stop_available"] = len(arr); stats["stop_used"] = int(n)

    probes = np.concatenate(picks, 0)                      # (n, 2) = (pid, entry)
    order = g.permutation(len(probes))
    probes = probes[order]
    slice_id = np.zeros(len(probes), np.int8)
    slice_id[:N_TUNE] = 1                                  # 1 = tuning, 0 = reporting
    bank_pids = g.choice(Q, size=BANK_PASSAGES, replace=False).astype(np.int32)

    qids, qtexts = load_queries(paths.QUERIES_TRAIN)
    sel = g.choice(len(qids), size=paths.N_QUERY_BANK_QUERIES, replace=False)
    bank_qidx = np.sort(sel).astype(np.int32)

    np.savez(p, probes=probes.astype(np.int64), slice_id=slice_id,
             bank_pids=bank_pids, bank_qidx=bank_qidx)
    save_json(stats, paths.RESULTS / "10_probe_plan.json")
    lg.info(f"probes={len(probes)} tuning={N_TUNE} bank_passages={len(bank_pids)} {stats}")
    return dict(np.load(p, allow_pickle=True))


def build_states(enc_name: str, force=False):
    """Pass 2: encode the planned texts and store word-unit states at all layers."""
    pp = paths.ART / f"probes_{enc_name}.npz"
    bp = paths.ART / f"bankH_{enc_name}.npy"
    bq = paths.ART / f"bankQ_{enc_name}.npy"
    if pp.exists() and bp.exists() and bq.exists() and not force:
        lg.info(f"{enc_name}: exists"); return
    pl = plan()
    col = Collection()
    enc = Encoder(enc_name)
    L = paths.LAYERS
    z = np.load(paths.ART / "vocab.npz")
    words = [str(w) for w in z["words"]]
    inv = {w: i for i, w in enumerate(words)}

    # ---- probes: first unit in the passage whose lowercase form is the entry ----
    probes = pl["probes"]
    upid, inv_idx = np.unique(probes[:, 0], return_inverse=True)
    want: dict[int, set] = {}
    for (pid, j), _ in zip(probes, range(len(probes))):
        want.setdefault(int(pid), set()).add(int(j))
    states = np.zeros((len(probes), len(L), enc.dim), np.float16)
    filled = np.zeros(len(probes), bool)
    by_pid: dict[int, list[int]] = {}
    for i, (pid, j) in enumerate(probes):
        by_pid.setdefault(int(pid), []).append(i)

    B = 256
    with Timer(f"{enc_name}: probe states", lg):
        for s in range(0, len(upid), B):
            pids = upid[s:s + B]
            texts = col.texts(pids)
            wu = enc.encode_word_units(texts, L, is_query=False,
                                       keep=lambda w: w in inv)
            st = wu.states.cpu().numpy()
            # first occurrence of each entry within each row
            seen = {}
            for u in range(len(wu.word)):
                key = (int(pids[wu.row[u]]), inv[wu.word[u].lower()])
                if key in seen:
                    continue
                seen[key] = u
            for pid in pids:
                for i in by_pid.get(int(pid), []):
                    u = seen.get((int(pid), int(probes[i, 1])))
                    if u is not None:
                        states[i] = st[u]
                        filled[i] = True
    lg.info(f"{enc_name}: probes filled {filled.mean():.4f}")
    np.savez(pp, states=states, probes=probes, slice_id=pl["slice_id"], filled=filled,
             layers=np.asarray(L))

    # ---- background bank B_H: all units of the bank passages, subsampled ----
    def bank_from(texts, is_query, out_path, tag):
        acc, tot = [], 0
        with Timer(f"{enc_name}: bank {tag}", lg):
            for s in range(0, len(texts), B):
                wu = enc.encode_word_units(texts[s:s + B], L, is_query=is_query)
                acc.append(wu.states.cpu().numpy())
                tot += len(wu.word)
                if tot >= paths.N_BANK * 1.15:
                    break
        A = np.concatenate(acc, 0)
        g = rng("bank", enc_name, tag)
        sel = g.choice(len(A), size=min(paths.N_BANK, len(A)), replace=False)
        A = A[np.sort(sel)]
        np.save(out_path, A)
        lg.info(f"{enc_name}: {tag} bank {A.shape} (from {tot} units)")

    bank_from(col.texts(pl["bank_pids"]), False, bp, "H")
    qids, qtexts = load_queries(paths.QUERIES_TRAIN)
    qsel = [qtexts[i] for i in pl["bank_qidx"]]
    bank_from(qsel, True, bq, "Q")


def build_aux(enc_name: str, force=False):
    """Word-unit states for the tau-fitting samples (§0.6): 5k S passages and dev queries."""
    sp = paths.ART / f"tauS_{enc_name}.npz"
    qp = paths.ART / f"tauQ_{enc_name}.npz"
    if sp.exists() and qp.exists() and not force:
        lg.info(f"{enc_name}: aux exists"); return
    col = Collection()
    S = load_splits()["S"]
    g = rng("tau-sample")
    pids = np.sort(g.choice(S, size=5000, replace=False))
    enc = Encoder(enc_name)
    L = paths.LAYERS

    def dump(texts, is_query, path, tag):
        rows, st, words = [], [], []
        off = 0
        B = 128
        with Timer(f"{enc_name}: tau sample {tag}", lg):
            for s0 in range(0, len(texts), B):
                wu = enc.encode_word_units(texts[s0:s0 + B], L, is_query=is_query)
                rows.append(wu.row.astype(np.int32) + s0)
                st.append(wu.states.cpu().numpy())
                words.extend(wu.word)
        np.savez(path, states=np.concatenate(st, 0), row=np.concatenate(rows, 0),
                 words=np.asarray(words), n_rows=np.int32(len(texts)),
                 layers=np.asarray(L))
        lg.info(f"{enc_name}: {tag} {np.concatenate(st,0).shape} over {len(texts)} texts")

    dump(col.texts(pids), False, sp, "S")
    np.save(paths.ART / "tauS_pids.npy", pids)
    _, qtexts = load_queries(paths.QUERIES_DEV_SMALL)
    dump(qtexts, True, qp, "devq")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--aux", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.encoder == "plan":
        plan(a.force)
    elif a.aux:
        build_aux(a.encoder, a.force)
    else:
        build_states(a.encoder, a.force)
