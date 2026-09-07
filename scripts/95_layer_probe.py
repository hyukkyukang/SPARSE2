"""Pilot M, layer selection: pick ONE hidden layer of a new backbone by training-free
retrieval, and fit its operating threshold.

The study's e5 layer was chosen by Pilot C's end-to-end retrieval after the §A.4 identity
rule picked the wrong one (identity peaks at layer 9, retrieval at 12; the two are
anti-correlated across layers). A 28-layer decoder has no obvious analogue of "the last
BERT layer", so guessing is unsafe and the identity rule is known not to work. This
applies the criterion that did work, at a fraction of C1's cost:

  probe corpus  every dev-small positive + N random passages of C1   (~107k passages)
  per layer     tau_d, tau_q from the §0.6 samples (targets 120 / 30 nnz)
                the untrained rule: whitened cosine, max over word units, ReLU(p - tau),
                log1p, top-128 / top-16; exact chunked scoring; MRR@10 and R@100
  choice        the layer with the best MRR@10; its tau_d is what 42_train.py needs

All candidate layers come out of ONE forward pass per batch (hidden_states), so five
layers cost one encode of the probe corpus. Deviation D14 in notes/deviations.md.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening, precision
from dvlsr.data import Collection, load_queries, qrels_dict
from dvlsr.encoders import Encoder
from dvlsr.sparse import profile_maxpool, find_tau
from dvlsr.util import get_logger, save_json, Timer, rng

lg = get_logger("layerprobe", "95_layer_probe.log")
DEV = "cuda"


def probe_pids(n_random, seed=11):
    """Encoder-independent, built once: dev-small positives + random C1 passages."""
    p = paths.PREP / f"probe_pids_{n_random}.npy"
    if p.exists():
        return np.load(p)
    qr = qrels_dict(paths.QRELS_DEV_SMALL)
    pos = {int(d) for docs in qr.values() for d in docs}
    c1 = np.load(paths.PREP / "c1_pids.npy")
    g = rng("layer-probe", seed)
    rand = g.choice(c1, size=min(n_random, len(c1)), replace=False)
    pids = np.unique(np.concatenate([np.fromiter(pos, np.int64), rand.astype(np.int64)]))
    tmp = p.with_suffix(f".tmp{os.getpid()}.npy")
    np.save(tmp, pids)
    os.replace(tmp, p)
    lg.info(f"probe corpus: {len(pos)} positives + {n_random} random -> {len(pids)} passages")
    return pids


def sparsify(P, tau, k):
    S = torch.log1p(torch.relu(P - tau))
    v, i = torch.topk(S, min(k, S.shape[1]), dim=1)
    return i, v


def score(qi, qv, di, dv, nV, k=100, dchunk=10000):
    Q = torch.zeros(qi.shape[0], nV, device=DEV)
    Q.scatter_(1, torch.as_tensor(qi, device=DEV, dtype=torch.long),
               torch.as_tensor(qv, device=DEV, dtype=torch.float32))
    best_v = torch.full((qi.shape[0], k), -1.0, device=DEV)
    best_i = torch.zeros((qi.shape[0], k), dtype=torch.long, device=DEV)
    for s in range(0, di.shape[0], dchunk):
        e = min(s + dchunk, di.shape[0])
        D = torch.zeros(e - s, nV, device=DEV)
        D.scatter_(1, torch.as_tensor(di[s:e], device=DEV, dtype=torch.long),
                   torch.as_tensor(dv[s:e], device=DEV, dtype=torch.float32))
        sc = Q @ D.T
        v, i = torch.topk(sc, min(k, e - s), dim=1)
        cv = torch.cat([best_v, v], 1); ci = torch.cat([best_i, i + s], 1)
        o = torch.topk(cv, k, dim=1).indices
        best_v, best_i = torch.gather(cv, 1, o), torch.gather(ci, 1, o)
        del D, sc
    return best_i.cpu().numpy()


def metrics(ranked, qids, qrels, loc):
    rr, rc = [], []
    for r, q in enumerate(qids):
        rel = {loc[int(d)] for d in qrels.get(str(q), {}) if int(d) in loc}
        if not rel:
            continue
        row = ranked[r]
        rr.append(next((1.0 / (i + 1) for i, d in enumerate(row[:10]) if int(d) in rel), 0.0))
        rc.append(len(rel & {int(x) for x in row[:100]}) / len(rel))
    return float(np.mean(rr)), float(np.mean(rc)), len(rr)


def main(a):
    enc_name = a.encoder
    L = a.layers or paths.layers_for(enc_name)
    Lall = paths.layers_for(enc_name)
    lidx = [Lall.index(l) for l in L]
    WD = paths.ART / "whiten"
    proto = np.load(paths.proto_file(enc_name))["protoA"]                 # (nV, |Lall|, d)
    nV = proto.shape[0]

    # entry matrices and token transforms, one per candidate layer
    E, tfH, tfQ = {}, {}, {}
    for l, li in zip(L, lidx):
        mu, W = whitening.load(WD / f"{enc_name}_L{l}_V_R2.npz")
        X = torch.as_tensor(np.asarray(proto[:, li], np.float32), device=DEV)
        E[l] = whitening.Transform("whitened", mu, W, DEV)(X).half()
        muH, WH = whitening.load(WD / f"{enc_name}_L{l}_H.npz")
        tfH[l] = whitening.Transform("whitened", muH, WH, DEV)
        muQ, WQ = whitening.load(WD / f"{enc_name}_L{l}_Q.npz")
        tfQ[l] = whitening.Transform("whitened", muQ, WQ, DEV)
    del proto

    # taus from the §0.6 samples (the same files Pilot C used)
    res = {}
    with Timer(f"{enc_name}: taus for layers {L}", lg), torch.no_grad(), precision.autocast():
        zS = np.load(paths.ART / f"tauS_{enc_name}.npz", allow_pickle=True)
        zQ = np.load(paths.ART / f"tauQ_{enc_name}.npz", allow_pickle=True)
        sS, sQ = zS["states"], zQ["states"]
        rowS = torch.as_tensor(zS["row"].astype(np.int64), device=DEV)
        rowQ = torch.as_tensor(zQ["row"].astype(np.int64), device=DEV)
        for l, li in zip(L, lidx):
            H = tfH[l](torch.as_tensor(np.asarray(sS[:, li], np.float32), device=DEV)).half()
            P = profile_maxpool(H, rowS, int(zS["n_rows"]), E[l])
            tau_d = find_tau(P, a.nnz_d)
            Hq = tfQ[l](torch.as_tensor(np.asarray(sQ[:, li], np.float32), device=DEV)).half()
            Pq = profile_maxpool(Hq, rowQ, int(zQ["n_rows"]), E[l])
            tau_q = find_tau(Pq, a.nnz_q)
            res[l] = dict(tau_d=tau_d, tau_q=tau_q,
                          nnz_d_sample=float((P > tau_d).sum(1).float().mean()),
                          nnz_q_sample=float((Pq > tau_q).sum(1).float().mean()))
            qi, qv = sparsify(Pq, tau_q, a.k_q)
            res[l]["_q"] = (qi.cpu().numpy(), qv.cpu().numpy())
            del H, P, Hq, Pq
            lg.info(f"{enc_name} L{l}: tau_d {tau_d:.4f} (nnz {res[l]['nnz_d_sample']:.0f}) "
                    f"tau_q {tau_q:.4f} (nnz {res[l]['nnz_q_sample']:.0f})")
        del sS, sQ
    torch.cuda.empty_cache()

    # encode the probe corpus ONCE for all candidate layers
    pids = probe_pids(a.n_random)
    loc = {int(p): i for i, p in enumerate(pids)}
    col = Collection()
    enc = Encoder(enc_name, dtype=torch.float16)
    n = len(pids)
    DI = {l: np.zeros((n, a.k_d), np.int32) for l in L}
    DV = {l: np.zeros((n, a.k_d), np.float16) for l in L}
    order = np.argsort([col.off[p + 1] - col.off[p] for p in pids])
    t0 = time.time()
    with torch.no_grad(), precision.autocast(), Timer(f"{enc_name}: encode probe corpus ({n})", lg):
        for s in range(0, n, a.bs):
            sel = order[s:s + a.bs]
            wu = enc.encode_word_units(col.texts(pids[sel]), L, is_query=False)
            if len(wu.word) == 0:
                continue
            row = torch.as_tensor(wu.row.astype(np.int64), device=DEV)
            for li_local, l in enumerate(L):
                h = tfH[l](wu.states[:, li_local].to(DEV)).half()
                P = profile_maxpool(h, row, len(sel), E[l])
                i, v = sparsify(P, res[l]["tau_d"], a.k_d)
                DI[l][sel] = i.cpu().numpy(); DV[l][sel] = v.cpu().numpy()
                del P
            if s % (a.bs * 200) == 0:
                lg.info(f"  {s}/{n}  {(s + a.bs) / max(time.time() - t0, 1e-9):.0f}/s")
    del enc
    torch.cuda.empty_cache()

    qids, _ = load_queries(paths.QUERIES_DEV_SMALL)
    qrels = qrels_dict(paths.QRELS_DEV_SMALL)
    for l in L:
        qi, qv = res[l].pop("_q")
        ranked = score(qi, qv, DI[l], DV[l], nV)
        mrr, r100, nq = metrics(ranked, qids, qrels, loc)
        res[l].update(mrr10=mrr, r100=r100, n_queries=nq,
                      nnz_d=float((DV[l] > 0).sum(1).mean()), nnz_q=float((qv > 0).sum(1).mean()))
        lg.info(f"{enc_name} L{l}: MRR@10 {mrr:.4f}  R@100 {r100:.4f}  nnz d/q {res[l]['nnz_d']:.0f}/{res[l]['nnz_q']:.0f}")
    best = max(L, key=lambda l: res[l]["mrr10"])
    out = dict(encoder=enc_name, layers=L, per_layer={str(l): res[l] for l in L},
               layer=int(best), tau=res[best]["tau_d"], tau_q=res[best]["tau_q"],
               n_probe=int(n), n_random=a.n_random)
    save_json(out, paths.RESULTS / f"95_layer_probe_{enc_name}.json")
    lg.info(f"{enc_name}: chosen layer {best} (MRR@10 {res[best]['mrr10']:.4f}), tau {res[best]['tau_d']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--n-random", type=int, default=100_000)
    ap.add_argument("--k-d", type=int, default=128)
    ap.add_argument("--k-q", type=int, default=16)
    ap.add_argument("--nnz-d", type=int, default=paths.NNZ_DOC_TARGET)
    ap.add_argument("--nnz-q", type=int, default=paths.NNZ_QRY_TARGET)
    ap.add_argument("--bs", type=int, default=64)
    main(ap.parse_args())
