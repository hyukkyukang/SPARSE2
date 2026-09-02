"""Pilot A — token-state geometry against a text-defined vocabulary (§A).

Sweeps (encoder x layer x transform x entry representation) and streams every
similarity profile: nothing of size |probes| x |V| is ever stored (§0.7).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.data import Collection
from dvlsr.encoders import Encoder
from dvlsr.sparse import profile_maxpool, find_tau
from dvlsr.util import get_logger, save_json, rng, Timer

lg = get_logger("pilotA", "14_pilotA.log")
WD = paths.ART / "whiten"
DEV = "cuda"
CHUNK = 4096


# ----------------------------------------------------------------- aux states
def aux_states(enc_name, force=False):
    """Word-unit states for the SPLADE-overlap passages and the polysemy occurrences."""
    p = paths.ART / f"aux2_{enc_name}.npz"
    if p.exists() and not force:
        return dict(np.load(p, allow_pickle=True))
    col = Collection()
    exp = np.load(paths.ART / "splade_expansions.npz", allow_pickle=True)
    poly = np.load(paths.ART / "polysemy.npz", allow_pickle=True)
    enc = Encoder(enc_name)
    L = paths.LAYERS
    out = {}

    ov = exp["overlap_pids"]
    rows, st = [], []
    for s in range(0, len(ov), 128):
        wu = enc.encode_word_units(col.texts(ov[s:s + 128]), L)
        rows.append(wu.row.astype(np.int32) + s); st.append(wu.states.cpu().numpy())
    out["ov_row"] = np.concatenate(rows); out["ov_states"] = np.concatenate(st, 0)
    out["ov_pids"] = ov

    z = np.load(paths.ART / "vocab.npz")
    inv = {str(w): i for i, w in enumerate(z["words"])}
    pw = [str(x) for x in poly["words"]]
    ppid = poly["pids"]
    states = np.zeros((len(ppid), len(L), enc.dim), np.float16)
    filled = np.zeros(len(ppid), bool)
    for s in range(0, len(ppid), 128):
        sl = slice(s, s + 128)
        wu = enc.encode_word_units(col.texts(ppid[sl]), L,
                                   keep=lambda w: w in inv)
        seen = {}
        for u in range(len(wu.word)):
            key = (int(wu.row[u]), wu.word[u].lower())
            seen.setdefault(key, u)
        for k, i in enumerate(range(s, min(s + 128, len(ppid)))):
            u = seen.get((k, pw[i]))
            if u is not None:
                states[i] = wu.states[u].cpu().numpy(); filled[i] = True
    out["poly_states"] = states; out["poly_filled"] = filled
    np.savez(p, **out)
    lg.info(f"{enc_name}: aux2 saved (overlap units {out['ov_states'].shape}, "
            f"polysemy filled {filled.mean():.3f})")
    return dict(np.load(p, allow_pickle=True))


# ----------------------------------------------------------------- transforms
def make_tf(mode, side_path):
    if mode == "raw":
        return whitening.Transform("raw")
    mu, W = whitening.load(side_path)
    return whitening.Transform(mode, mu, W, DEV)


def entry_matrix(rep, enc_name, li, r1_prefix, r1_shared=False):
    if rep == "R2":
        E = np.load(paths.ART / f"proto_{enc_name}.npz")["protoA"][:, li]
        wpath = WD / f"{enc_name}_L{paths.LAYERS[li]}_V_R2.npz"
    else:
        E = np.load(paths.ART / f"r1_{enc_name}.npz")[r1_prefix]
        wpath = WD / f"{enc_name}_R1{r1_prefix}_V.npz"
    if r1_shared:
        wpath = WD / f"{enc_name}_L{paths.LAYERS[li]}_H.npz"
    return np.asarray(E, np.float32), wpath


# ----------------------------------------------------------------- metrics
def profile_chunks(H, E, chunk=CHUNK):
    for s in range(0, H.shape[0], chunk):
        yield s, (H[s:s + chunk] @ E.T).float()


def probe_metrics(H, own, E, tau, rel_map=None, n_rand=4, seed=0):
    """Streamed probe-level metrics. H:(n,d) normalised, E:(|V|,d) normalised."""
    n, nV = H.shape[0], E.shape[0]
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    acc = dict(rank=torch.zeros(n, device=DEV), self_sim=torch.zeros(n, device=DEV),
               zgap=torch.zeros(n, device=DEV), pr=torch.zeros(n, device=DEV),
               nnz=torch.zeros(n, device=DEV), auc=torch.zeros(n, device=DEV))
    rel_rr, rel_n, rnd_rr = torch.zeros(n, device=DEV), torch.zeros(n, device=DEV), torch.zeros(n, device=DEV)
    for s, A in profile_chunks(H, E):
        m = A.shape[0]
        idx = torch.arange(m, device=DEV)
        o = own[s:s + m]
        ss = A[idx, o]
        acc["self_sim"][s:s + m] = ss
        acc["rank"][s:s + m] = (A > ss.unsqueeze(1)).sum(1).float()
        med = A.median(dim=1).values
        sd = A.std(dim=1)
        acc["zgap"][s:s + m] = (A.max(dim=1).values - med) / sd.clamp(min=1e-6)
        r = torch.relu(A - tau)
        acc["pr"][s:s + m] = (r.sum(1) ** 2) / (r.pow(2).sum(1).clamp(min=1e-12))
        acc["nnz"][s:s + m] = (A > tau).sum(1).float()
        rnd = torch.randint(0, nV, (m, n_rand), device=DEV, generator=g)
        acc["auc"][s:s + m] = (ss.unsqueeze(1) > A.gather(1, rnd)).float().mean(1)
        if rel_map is not None:
            for k in range(m):
                terms = rel_map.get(int(s + k))
                if not terms:
                    continue
                t = torch.tensor(terms, device=DEV)
                rk = (A[k].unsqueeze(0) > A[k][t].unsqueeze(1)).sum(1).float() + 1
                rel_rr[s + k] = (1.0 / rk).mean()
                rel_n[s + k] = len(terms)
                rt = torch.randint(0, nV, (len(terms),), device=DEV, generator=g)
                rk2 = (A[k].unsqueeze(0) > A[k][rt].unsqueeze(1)).sum(1).float() + 1
                rnd_rr[s + k] = (1.0 / rk2).mean()
        del A
    out = {k: v.cpu().numpy() for k, v in acc.items()}
    out["rel_rr"] = rel_rr.cpu().numpy(); out["rel_n"] = rel_n.cpu().numpy()
    out["rnd_rr"] = rnd_rr.cpu().numpy()
    return out


def summarize(m, is_stop, decile, mask=None):
    def agg(sel):
        if sel.sum() == 0:
            return {}
        return dict(
            self_hit10=float((m["rank"][sel] < 10).mean()),
            self_hit1=float((m["rank"][sel] < 1).mean()),
            self_mrr=float((1.0 / (m["rank"][sel] + 1)).mean()),
            auc=float(m["auc"][sel].mean()),
            zgap=float(m["zgap"][sel].mean()),
            pr=float(m["pr"][sel].mean()),
            nnz_tok=float(m["nnz"][sel].mean()),
            n=int(sel.sum()))
    base = np.ones(len(is_stop), bool) if mask is None else mask.copy()
    out = dict(all=agg(base), content=agg(base & ~is_stop), stop=agg(base & is_stop))
    out["by_decile"] = {int(d): agg(base & ~is_stop & (decile == d)) for d in range(10)}
    sel = base & (m["rel_n"] > 0)
    if sel.sum() > 0:
        out["related_mrr"] = float(m["rel_rr"][sel].mean())
        out["related_mrr_random"] = float(m["rnd_rr"][sel].mean())
        out["related_ratio"] = out["related_mrr"] / max(out["related_mrr_random"], 1e-12)
        out["related_n_probes"] = int(sel.sum())
    return out


def sense_accuracy(states_li, poly, tfH, E, anchors):
    """margin = mean sim to correct-sense anchors - mean to wrong-sense anchors."""
    H = tfH(torch.as_tensor(states_li, device=DEV))
    words = [str(x) for x in poly["words"]]
    senses = [str(x) for x in poly["senses"]]
    keys = [str(x) for x in poly["anchor_keys"]]
    aid = {k: poly["anchor_ids"][i] for i, k in enumerate(keys)}
    filled = poly["filled"] if "filled" in poly else np.ones(len(words), bool)
    A = (H @ E.T).float()
    ok, correct = 0, 0
    per_word = {}
    for i in range(len(words)):
        if not filled[i]:
            continue
        w, s = words[i], senses[i]
        other = "B" if s == "A" else "A"
        c = A[i, torch.as_tensor(aid[f"{w}|{s}"], device=DEV, dtype=torch.long)].mean()
        d = A[i, torch.as_tensor(aid[f"{w}|{other}"], device=DEV, dtype=torch.long)].mean()
        hit = bool((c - d) > 0)
        ok += 1; correct += hit
        pw = per_word.setdefault(w, [0, 0])
        pw[0] += hit; pw[1] += 1
    return dict(accuracy=correct / max(ok, 1), n=ok,
                per_word={k: v[0] / v[1] for k, v in per_word.items()})


def splade_overlap(aux, exp, tfH, E, tau, vocab_words, inv, splade_vocab, topn=20):
    row = torch.as_tensor(aux["ov_row"].astype(np.int64), device=DEV)
    H = tfH(torch.as_tensor(aux["ov_states"], device=DEV))
    n_rows = len(aux["ov_pids"])
    P = profile_maxpool(H, row, n_rows, E)
    ours = torch.topk(torch.relu(P - tau), topn, dim=1).indices.cpu().numpy()
    pid2i = {int(p): i for i, p in enumerate(exp["pids"])}
    sp_vocab = [str(x) for x in splade_vocab]
    in_splade = np.array([w in set(sp_vocab) for w in vocab_words])
    jac, cov = [], []
    for i, pid in enumerate(aux["ov_pids"]):
        e = pid2i[int(pid)]
        toks = [sp_vocab[t] for t in exp["idx"][e][:topn]]
        S = {inv[t] for t in toks if t in inv}
        O = {int(j) for j in ours[i] if in_splade[j]}
        cov.append(len(S) / topn)
        if S or O:
            jac.append(len(S & O) / max(len(S | O), 1))
    return dict(jaccard=float(np.mean(jac)), coverage=float(np.mean(cov)), n=len(jac))


# ----------------------------------------------------------------- main sweep
def run(enc_name, layers=None, reps=("R1", "R2"), transforms=("raw", "centered", "whitened"),
        do_sense=True, do_overlap=True, r1_prefix=None, tag=""):
    z = np.load(paths.ART / "vocab.npz")
    words = [str(w) for w in z["words"]]
    inv = {w: i for i, w in enumerate(words)}
    is_stop_v, decile_v = z["is_stop"], z["decile"]
    pr = np.load(paths.ART / f"probes_{enc_name}.npz", allow_pickle=True)
    probes, slice_id, filled = pr["probes"], pr["slice_id"], pr["filled"]
    L = list(paths.LAYERS)
    layers = layers or L
    tauS = np.load(paths.ART / f"tauS_{enc_name}.npz", allow_pickle=True)
    exp = np.load(paths.ART / "splade_expansions.npz", allow_pickle=True)
    poly = np.load(paths.ART / "polysemy.npz", allow_pickle=True)
    aux = aux_states(enc_name)
    poly = dict(poly); poly["filled"] = aux["poly_filled"]

    # related-term map: probe index -> entry ids of its passage's SPLADE expansions
    pid2e = {int(p): i for i, p in enumerate(exp["pids"])}
    sp_vocab = [str(x) for x in exp["vocab"]]
    rel_all = {}
    for i, (pid, j) in enumerate(probes):
        e = pid2e.get(int(pid))
        if e is None:
            continue
        terms = []
        for t in exp["idx"][e]:
            w = sp_vocab[t]
            k = inv.get(w)
            if k is not None and k != int(j):
                terms.append(k)
            if len(terms) == 10:
                break
        if terms:
            rel_all[i] = terms

    results = {}
    for slice_name, sel_mask in (("tuning", slice_id == 1), ("reporting", slice_id == 0)):
        if tag == "prefix" and slice_name == "reporting":
            continue
        if tag != "prefix" and slice_name == "tuning":
            continue
        keep = sel_mask & filled
        idx = np.flatnonzero(keep)
        own = torch.as_tensor(probes[idx, 1], device=DEV, dtype=torch.long)
        pstop = is_stop_v[probes[idx, 1]]
        pdec = decile_v[probes[idx, 1]]
        remap = {int(k): v for k, v in
                 ((np.searchsorted(idx, i), rel_all[i]) for i in rel_all
                  if i in set(idx.tolist()))} if False else None
        pos = {int(v): k for k, v in enumerate(idx)}
        remap = {pos[i]: t for i, t in rel_all.items() if i in pos}

        for li, l in [(L.index(x), x) for x in layers]:
            Hraw = torch.as_tensor(pr["states"][idx, li], device=DEV)
            for rep in reps:
                prefixes = [r1_prefix or "doc"] if rep == "R1" else [None]
                if tag == "prefix" and rep == "R1":
                    prefixes = list(np.load(paths.ART / f"r1_{enc_name}.npz").files)
                for pfx in prefixes:
                    shared_opts = [False, True] if (rep == "R1" and tag != "prefix") else [False]
                    for shared in shared_opts:
                        for tr in transforms:
                            E_np, wpath = entry_matrix(rep, enc_name, li, pfx, shared)
                            tfV = make_tf(tr, wpath)
                            tfH = make_tf(tr, WD / f"{enc_name}_L{l}_H.npz")
                            E = tfV(torch.as_tensor(E_np, device=DEV)).half()
                            H = tfH(Hraw).half()
                            # tau at the §0.6 operating point for this space
                            Hs = tfH(torch.as_tensor(tauS["states"][:, li], device=DEV)).half()
                            rowS = torch.as_tensor(tauS["row"].astype(np.int64), device=DEV)
                            PS = profile_maxpool(Hs, rowS, int(tauS["n_rows"]), E)
                            tau = find_tau(PS, paths.NNZ_DOC_TARGET)
                            tok_pct = float((PS.reshape(-1) < tau).float().mean() * 100)
                            del PS, Hs
                            m = probe_metrics(H, own, E, tau, remap)
                            key = f"L{l}|{rep}{'-shared' if shared else ''}" \
                                  f"{'-' + pfx if pfx else ''}|{tr}"
                            r = summarize(m, pstop, pdec)
                            r["tau"] = tau
                            r["tau_token_percentile"] = tok_pct
                            if do_sense:
                                r["sense"] = sense_accuracy(
                                    aux["poly_states"][:, li], poly, tfH, E, None)
                            if do_overlap:
                                r["splade_overlap"] = splade_overlap(
                                    aux, exp, tfH, E, tau, words, inv, exp["vocab"])
                            results[key] = r
                            lg.info(f"{enc_name} {key}: hit10={r['content']['self_hit10']:.3f} "
                                    f"(stop {r['stop']['self_hit10']:.3f}) "
                                    f"relMRR={r.get('related_mrr', 0):.4f} "
                                    f"z={r['content']['zgap']:.2f} "
                                    f"nnz/tok={r['content']['nnz_tok']:.1f} "
                                    + (f"sense={r['sense']['accuracy']:.3f} " if do_sense else "")
                                    + (f"jac={r['splade_overlap']['jaccard']:.3f}" if do_overlap else ""))
                            del E, H
                            torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--tag", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--reps", default="R1,R2")
    ap.add_argument("--r1-prefix", default=None)
    ap.add_argument("--no-sense", action="store_true")
    ap.add_argument("--no-overlap", action="store_true")
    a = ap.parse_args()
    ls = [int(x) for x in a.layers.split(",")] if a.layers else None
    with Timer(f"pilot A {a.encoder} {a.tag}", lg):
        res = run(a.encoder, ls, tuple(a.reps.split(",")), do_sense=not a.no_sense,
                  do_overlap=not a.no_overlap, r1_prefix=a.r1_prefix, tag=a.tag)
    out = paths.RESULTS / f"14_pilotA_{a.encoder}{'_' + a.tag if a.tag else ''}.json"
    save_json(res, out)
    lg.info(f"saved {out}")
