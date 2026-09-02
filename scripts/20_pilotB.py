"""Pilot B — what a vocabulary entry *is*: bare string vs contextual prototype (§B).

Decides the deployment cost of adding an entry (one forward pass vs k occurrences)
and measures the calibration risk (hubness, runaway posting lists) that Pilot D
then has to survive.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.data import Collection, load_splits
from dvlsr.encoders import Encoder
from dvlsr.sparse import profile_maxpool, find_tau
from dvlsr.util import get_logger, save_json, rng, Timer

lg = get_logger("pilotB", "20_pilotB.log")
DEV = "cuda"
WD = paths.ART / "whiten"
K_CURVE = [1, 3, 5, 10, 20, 50]
TOPK_DOC = 256


# ------------------------------------------------------------------ builders
def occ_store(enc, layers):
    return np.load(paths.ART / f"occ_{enc}_L{'-'.join(map(str, layers))}.npy", mmap_mode="r")


def build_reps(enc, layer, layers_in_store, r1_prefix, want_r3=False):
    """Every entry representation of §B.1, as raw (unwhitened) float32 matrices."""
    li = layers_in_store.index(layer)
    occ = occ_store(enc, layers_in_store)
    z = np.load(paths.ART / "vocab.npz")
    occ_n = z["occ_n"]
    reps = {}
    r1 = np.load(paths.ART / f"r1_{enc}.npz")[r1_prefix].astype(np.float32)
    reps["R1"] = r1
    # k-curve on two disjoint occurrence sets
    for half, base in (("S1", 0), ("S2", 50)):
        run = None
        for k in K_CURVE:
            blk = np.asarray(occ[:, base:base + k, li], np.float32)
            reps[f"R2-{half}-k{k}"] = blk.mean(1)
    reps["R2"] = reps["R2-S1-k50"]
    reps["R2p"] = reps["R2-S2-k50"]
    if want_r3:
        from sklearn.cluster import KMeans
        cents = np.zeros((len(r1), 5, r1.shape[1]), np.float32)
        for j in range(len(r1)):
            X = np.asarray(occ[j, : int(occ_n[j]), li], np.float32)
            k = min(5, len(X))
            km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(X)
            c = km.cluster_centers_
            cents[j, :len(c)] = c
            if len(c) < 5:
                cents[j, len(c):] = c[0]
        reps["R3"] = cents                       # (|V|, 5, d) -> max over centroids
    return reps


def whiten_rep(name, X, enc, layer, shared=False):
    """§0.5: one transform per representation, from that representation's own matrix."""
    if shared:
        mu, W = whitening.load(WD / f"{enc}_L{layer}_H.npz")
    else:
        flat = X.reshape(-1, X.shape[-1])
        mu, W, _ = whitening.estimate(flat)
    t = whitening.Transform("whitened", mu, W, DEV)
    return t(torch.as_tensor(X, device=DEV).reshape(-1, X.shape[-1])).reshape(
        *X.shape).half()


def token_tf(enc, layer):
    mu, W = whitening.load(WD / f"{enc}_L{layer}_H.npz")
    return whitening.Transform("whitened", mu, W, DEV)


def sim_to_entries(H, E):
    """cos to entries; E may be (|V|,d) or (|V|,m,d) for multi-prototype (max over m)."""
    if E.dim() == 2:
        return (H @ E.T).float()
    n, m, d = E.shape
    A = (H @ E.reshape(-1, d).T).float().reshape(H.shape[0], n, m)
    return A.max(-1).values


# ------------------------------------------------------------------ metrics
def hubness(E, bank, tf, topn=10, chunk=8192):
    n_entries = E.shape[0]
    counts = torch.zeros(n_entries, device=DEV)
    total = 0
    for s in range(0, bank.shape[0], chunk):
        H = tf(torch.as_tensor(bank[s:s + chunk], device=DEV)).half()
        A = sim_to_entries(H, E)
        idx = torch.topk(A, topn, dim=1).indices.reshape(-1)
        counts.index_add_(0, idx, torch.ones(len(idx), device=DEV))
        total += A.shape[0] * topn
        del A, H
    c = counts.cpu().numpy()
    order = np.argsort(-c)
    top1pct = max(1, n_entries // 100)
    return dict(skew=float(sps.skew(c)), hub_share=float(c[order[:top1pct]].sum() / total),
                max_count=float(c.max()), gini=float(_gini(c))), c


def _gini(x):
    x = np.sort(np.asarray(x, np.float64))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum() + 1e-12))


def self_activation(E, enc_obj, tf, words, layer_idx, layers, bs=512):
    """Rank of entry j in the profile of its own word given as the input text."""
    ranks = np.zeros(len(words), np.int64)
    for s in range(0, len(words), bs):
        chunk = words[s:s + bs]
        wu = enc_obj.encode_word_units(chunk, layers)
        H = tf(wu.states[:, layer_idx].to(DEV)).half()
        row = torch.as_tensor(wu.row.astype(np.int64), device=DEV)
        P = profile_maxpool(H, row, len(chunk), E) if E.dim() == 2 else None
        if P is None:                       # multi-prototype: pool manually
            A = sim_to_entries(H, E)
            P = torch.full((len(chunk), E.shape[0]), -1.0, device=DEV)
            P.scatter_reduce_(0, row.unsqueeze(1).expand(-1, A.shape[1]), A, reduce="amax")
        own = torch.arange(s, min(s + bs, len(words)), device=DEV)
        v = P[torch.arange(P.shape[0], device=DEV), own]
        ranks[s:s + len(chunk)] = (P > v.unsqueeze(1)).sum(1).cpu().numpy()
        del P
    return dict(median_rank=float(np.median(ranks) + 1),
                p_rank1=float((ranks == 0).mean()),
                p_top10=float((ranks < 10).mean()))


def stability(reps, enc, layer, bank, tf, topn=50, chunk=2048):
    """Top-50 nearest bank tokens of the S1- and S2-prototypes, per k (§B.3)."""
    Hb = tf(torch.as_tensor(bank[:50000], device=DEV)).half()
    out = {}
    for k in K_CURVE:
        nn = {}
        for half in ("S1", "S2"):
            E = whiten_rep(f"R2-{half}-k{k}", reps[f"R2-{half}-k{k}"], enc, layer)
            top = []
            for s in range(0, E.shape[0], chunk):
                A = (E[s:s + chunk] @ Hb.T).float()
                top.append(torch.topk(A, topn, dim=1).indices.cpu().numpy())
                del A
            nn[half] = np.concatenate(top, 0)
            nn[half + "_E"] = E
        j = np.array([len(np.intersect1d(a, b)) / (2 * topn - len(np.intersect1d(a, b)))
                      for a, b in zip(nn["S1"], nn["S2"])])
        cos = torch.nn.functional.cosine_similarity(
            nn["S1_E"].float(), nn["S2_E"].float(), dim=-1).cpu().numpy()
        out[k] = dict(jaccard=float(j.mean()), whitened_cos=float(cos.mean()))
        lg.info(f"stability k={k}: jaccard={j.mean():.3f} cos={cos.mean():.3f}")
    return out


def df_pass(reps_w, enc_obj, tf, layer_idx, layers, taus, n_docs=None, bs=192):
    """One streaming pass over S: document frequency and weight statistics per entry (§B.2.3)."""
    col = Collection()
    S = load_splits()["S"]
    if n_docs:
        S = S[:n_docs]
    nV = next(iter(reps_w.values())).shape[0]
    df = {k: torch.zeros(nV, device=DEV) for k in reps_w}
    wsum = {k: torch.zeros(nV, device=DEV) for k in reps_w}
    nnz = {k: 0.0 for k in reps_w}
    for s in range(0, len(S), bs):
        pids = S[s:s + bs]
        wu = enc_obj.encode_word_units(col.texts(pids), layers)
        if len(wu.word) == 0:
            continue
        H = tf(wu.states[:, layer_idx].to(DEV)).half()
        row = torch.as_tensor(wu.row.astype(np.int64), device=DEV)
        for k, E in reps_w.items():
            P = profile_maxpool(H, row, len(pids), E)
            V = torch.relu(P - taus[k])
            val, idx = torch.topk(V, min(TOPK_DOC, nV), dim=1)
            m = val > 0
            df[k].index_add_(0, idx[m], torch.ones(int(m.sum()), device=DEV))
            wsum[k].index_add_(0, idx[m], val[m].float())
            nnz[k] += float(m.sum())
            del P, V
        del H
    n = len(S)
    return {k: dict(df=df[k].cpu().numpy(), wsum=wsum[k].cpu().numpy(),
                    nnz_per_doc=nnz[k] / n, n_docs=n) for k in reps_w}


def main(enc, layer, layers_in_store, r1_prefix, want_r3, n_docs, tag=""):
    z = np.load(paths.ART / "vocab.npz")
    words = [str(w) for w in z["words"]]
    freq, decile, is_stop = z["freq"], z["decile"], z["is_stop"]
    inv = {w: i for i, w in enumerate(words)}
    bank = np.load(paths.ART / f"bankH_{enc}.npy", mmap_mode="r")
    li_bank = paths.LAYERS.index(layer)
    bankL = np.asarray(bank[:, li_bank])
    tf = token_tf(enc, layer)
    enc_obj = Encoder(enc)
    li_store = layers_in_store.index(layer)

    with Timer(f"build representations (layer {layer})", lg):
        reps = build_reps(enc, layer, layers_in_store, r1_prefix, want_r3)

    main_reps = ["R1", "R1-shared", "R2"] + (["R3"] if want_r3 else [])
    reps_w = {}
    for name in main_reps:
        base = reps["R1"] if name.startswith("R1") else reps[name]
        reps_w[name] = whiten_rep(name, base, enc, layer, shared=name.endswith("-shared"))

    # tau re-fit per representation to a common nnz target (§B.2.3 density control)
    tauS = np.load(paths.ART / f"tauS_{enc}.npz", allow_pickle=True)
    li_tau = list(tauS["layers"]).index(layer)
    Hs = tf(torch.as_tensor(tauS["states"][:, li_tau], device=DEV)).half()
    rowS = torch.as_tensor(tauS["row"].astype(np.int64), device=DEV)
    taus = {}
    for name, E in reps_w.items():
        PS = profile_maxpool(Hs, rowS, int(tauS["n_rows"]), E)
        taus[name] = find_tau(PS, paths.NNZ_DOC_TARGET)
        del PS
    lg.info(f"taus: { {k: round(v,4) for k,v in taus.items()} }")
    del Hs

    res = {"layer": layer, "encoder": enc, "r1_prefix": r1_prefix, "taus": taus}

    # ---- hubness -----------------------------------------------------------
    hubs = {}
    for name, E in reps_w.items():
        with Timer(f"hubness {name}", lg):
            h, counts = hubness(E, bankL, tf)
        order = np.argsort(-counts)[:20]
        h["top20_hubs"] = [[words[int(i)], int(counts[i]), int(decile[i]),
                            bool(is_stop[i])] for i in order]
        hubs[name] = h
        lg.info(f"hubness {name}: skew={h['skew']:.1f} share={h['hub_share']:.3f} "
                f"top: {[x[0] for x in h['top20_hubs'][:8]]}")
    res["hubness"] = hubs

    # ---- df calibration ----------------------------------------------------
    with Timer("df pass over S", lg):
        dfs = df_pass(reps_w, enc_obj, tf, li_tau, list(tauS["layers"]), taus, n_docs)
    dfres = {}
    for name, d in dfs.items():
        df, n = d["df"], d["n_docs"]
        rho = float(sps.spearmanr(df, freq).statistic)
        runaway = float((df > 0.2 * n).mean())
        wbar = d["wsum"] / np.maximum(df, 1)
        dfres[name] = dict(
            spearman_df_freq=rho, runaway_rate=runaway, nnz_per_doc=d["nnz_per_doc"],
            n_docs=n, dead_rate=float((df == 0).mean()),
            df_by_decile={int(dd): float(np.median(df[decile == dd])) for dd in range(10)},
            wbar_by_decile={int(dd): float(np.median(wbar[decile == dd])) for dd in range(10)},
            df_median=float(np.median(df)), df_p99=float(np.percentile(df, 99)))
        lg.info(f"df {name}: spearman={rho:.3f} runaway={runaway:.4f} "
                f"dead={dfres[name]['dead_rate']:.3f} nnz/doc={d['nnz_per_doc']:.1f}")
        np.save(paths.ART / f"pilotB_df_{enc}_L{layer}_{name}.npy", df)
    res["df"] = dfres

    # ---- in-context self-hit + sense ---------------------------------------
    from dvlsr.probes import probe_metrics, summarize, sense_accuracy
    pr = np.load(paths.ART / f"probes_{enc}.npz", allow_pickle=True)
    keep = (pr["slice_id"] == 0) & pr["filled"]
    idx = np.flatnonzero(keep)
    li_probe = list(pr["layers"]).index(layer)
    H = tf(torch.as_tensor(pr["states"][idx, li_probe], device=DEV)).half()
    own = torch.as_tensor(pr["probes"][idx, 1], device=DEV, dtype=torch.long)
    pstop, pdec = is_stop[pr["probes"][idx, 1]], decile[pr["probes"][idx, 1]]
    aux = dict(np.load(paths.ART / f"aux2_{enc}.npz", allow_pickle=True))
    poly = dict(np.load(paths.ART / "polysemy.npz", allow_pickle=True))
    poly["filled"] = aux["poly_filled"]
    li_aux = paths.LAYERS.index(layer)
    ic = {}
    for name, E in reps_w.items():
        m = probe_metrics(H, own, E, taus[name])
        s = summarize(m, pstop, pdec)
        s["sense"] = sense_accuracy(aux["poly_states"][:, li_aux], poly, tf, E)
        s["self_activation"] = self_activation(E, enc_obj, tf, words, li_aux,
                                               paths.LAYERS)
        ic[name] = s
        lg.info(f"in-context {name}: hit10={s['content']['self_hit10']:.3f} "
                f"sense={s['sense']['accuracy']:.3f} "
                f"selfact_rank={s['self_activation']['median_rank']:.0f} "
                f"P(rank1)={s['self_activation']['p_rank1']:.3f}")
    res["in_context"] = ic
    del H

    # ---- stability curve ---------------------------------------------------
    with Timer("stability curve", lg):
        res["stability"] = stability(reps, enc, layer, bankL, tf)
    kstar = None
    ref = res["stability"][50]["jaccard"]
    for k in K_CURVE:
        if res["stability"][k]["jaccard"] >= 0.8 * ref:
            kstar = k
            break
    res["k_star"] = kstar
    lg.info(f"k* (smallest k reaching 0.8 x k=50 Jaccard) = {kstar}")

    out = paths.RESULTS / f"20_pilotB_{enc}_L{layer}{tag}.json"
    save_json(res, out)
    lg.info(f"saved {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--store-layers", required=True)
    ap.add_argument("--r1-prefix", default="doc")
    ap.add_argument("--r3", action="store_true")
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.encoder, a.layer, [int(x) for x in a.store_layers.split(",")],
         a.r1_prefix, a.r3, a.n_docs, a.tag)
