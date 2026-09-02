"""Probe-level metrics shared by Pilots A and B (§A.3, §B.3)."""
from __future__ import annotations
import numpy as np
import torch

from .sparse import profile_maxpool
from .util import rng

DEV = "cuda"
CHUNK = 4096


def entry_sim(H, E):
    """cos of token states to entries; E is (|V|,d), or (|V|,m,d) for multi-prototype."""
    if E.dim() == 2:
        return (H @ E.T).float()
    n, m, d = E.shape
    return (H @ E.reshape(-1, d).T).float().reshape(H.shape[0], n, m).max(-1).values


def profile_chunks(H, E, chunk=CHUNK):
    for s in range(0, H.shape[0], chunk):
        yield s, entry_sim(H[s:s + chunk], E)


def probe_metrics(H, own, E, tau, T=None, M=None, R=None, n_rand=4, seed=0):
    """Streamed probe-level metrics. H:(n,d) normalised, E:(|V|,d) normalised.

    T/M/R: padded related-term ids, their mask, and matched random ids (n, 10).
    """
    n, nV = H.shape[0], E.shape[0]
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    keys = ("rank", "self_sim", "zgap", "pr", "nnz", "auc", "rel_rr", "rnd_rr")
    acc = {k: torch.zeros(n, device=DEV) for k in keys}
    for s, A in profile_chunks(H, E):
        m = A.shape[0]
        idx = torch.arange(m, device=DEV)
        ss = A[idx, own[s:s + m]]
        acc["self_sim"][s:s + m] = ss
        acc["rank"][s:s + m] = (A > ss.unsqueeze(1)).sum(1).float()
        acc["zgap"][s:s + m] = ((A.max(dim=1).values - A.median(dim=1).values)
                                / A.std(dim=1).clamp(min=1e-6))
        r = torch.relu(A - tau)
        acc["pr"][s:s + m] = (r.sum(1) ** 2) / (r.pow(2).sum(1).clamp(min=1e-12))
        acc["nnz"][s:s + m] = (A > tau).sum(1).float()
        rnd = torch.randint(0, nV, (m, n_rand), device=DEV, generator=g)
        acc["auc"][s:s + m] = (ss.unsqueeze(1) > A.gather(1, rnd)).float().mean(1)
        if T is not None:
            for tag, ids in (("rel_rr", T[s:s + m]), ("rnd_rr", R[s:s + m])):
                vals = A.gather(1, ids.clamp(min=0))
                rk = torch.stack([(A > vals[:, c:c + 1]).sum(1) + 1
                                  for c in range(ids.shape[1])], 1).float()
                mk = M[s:s + m].float()
                acc[tag][s:s + m] = ((1.0 / rk) * mk).sum(1) / mk.sum(1).clamp(min=1)
        del A
    out = {k: v.cpu().numpy() for k, v in acc.items()}
    out["rel_n"] = (M.sum(1).cpu().numpy() if M is not None else np.zeros(n))
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


def sense_accuracy(states_li, poly, tfH, E):
    """margin = mean sim to correct-sense anchors - mean to wrong-sense anchors."""
    H = tfH(torch.as_tensor(states_li, device=DEV)).to(E.dtype)
    words = [str(x) for x in poly["words"]]
    senses = [str(x) for x in poly["senses"]]
    keys = [str(x) for x in poly["anchor_keys"]]
    aid = {k: poly["anchor_ids"][i] for i, k in enumerate(keys)}
    filled = poly["filled"] if "filled" in poly else np.ones(len(words), bool)
    A = entry_sim(H, E)
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


def splade_overlap(aux, exp, tfH, E, tau, vocab_words, inv, splade_vocab, li, topn=20):
    row = torch.as_tensor(aux["ov_row"].astype(np.int64), device=DEV)
    H = tfH(torch.as_tensor(aux["ov_states"][:, li], device=DEV)).to(E.dtype)
    n_rows = len(aux["ov_pids"])
    P = profile_maxpool(H, row, n_rows, E)
    ours = torch.topk(torch.relu(P - tau), topn, dim=1).indices.cpu().numpy()
    pid2i = {int(p): i for i, p in enumerate(exp["pids"])}
    sp_vocab = [str(x) for x in splade_vocab]
    sp_set = set(sp_vocab)
    in_splade = np.array([w in sp_set for w in vocab_words])
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


