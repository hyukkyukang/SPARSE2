"""Exact sparse retrieval over an inverted index held on the GPU.

The document store is the top-N (entry, p) pairs per document, already sorted by
p descending, so a grid cell (tau, k_d, saturation) is a *mask* over the same
postings rather than a re-encoding (§C.1). Scoring accumulates
    score(q, d) = sum_j s_j(q) * s_j(d)
by expanding each query's posting lists and scattering into a dense score row --
fully vectorised, so 36 grid cells cost minutes, not hours.
"""
from __future__ import annotations
import numpy as np
import torch


def saturate(x, mode):
    return torch.log1p(x) if mode == "log1p" else x


class InvertedIndex:
    """Postings grouped by entry, with the per-document rank kept for k_d masking."""

    def __init__(self, idx: np.ndarray, val: np.ndarray, n_entries: int,
                 device="cuda", dtype=torch.float32, min_val: float = -1e3):
        n_docs, top = idx.shape
        e = torch.as_tensor(idx.reshape(-1).astype(np.int64))
        d = torch.arange(n_docs, dtype=torch.int64).repeat_interleave(top)
        p = torch.as_tensor(val.reshape(-1).astype(np.float32))
        r = torch.arange(top, dtype=torch.int16).repeat(n_docs)
        keep = p > min_val
        e, d, p, r = e[keep], d[keep], p[keep], r[keep]
        order = torch.argsort(e)
        self.doc = d[order].to(device=device, dtype=torch.int32)
        self.p = p[order].to(device=device, dtype=dtype)
        self.rank = r[order].to(device=device)
        counts = torch.bincount(e, minlength=n_entries)
        self.ptr = torch.zeros(n_entries + 1, dtype=torch.int64, device=device)
        self.ptr[1:] = torch.cumsum(counts, 0).to(device)
        self.n_docs, self.n_entries = n_docs, n_entries
        self.device = device

    def df(self, tau: float, k_d: int) -> torch.Tensor:
        m = (self.p > tau) & (self.rank < k_d)
        e = torch.repeat_interleave(
            torch.arange(self.n_entries, device=self.device),
            (self.ptr[1:] - self.ptr[:-1]))
        return torch.bincount(e[m], minlength=self.n_entries)

    def search(self, q_idx: np.ndarray, q_val: np.ndarray, tau_q: float, tau_d: float,
               k_q: int, k_d: int, saturation: str = "none", k: int = 1000,
               qchunk: int = 128, budget: int = 60_000_000):
        """Top-k documents per query. q_idx/q_val are the stored top-N query profiles."""
        nq = q_idx.shape[0]
        out_i = np.zeros((nq, k), np.int32)
        out_s = np.zeros((nq, k), np.float32)
        qi = torch.as_tensor(q_idx.astype(np.int64), device=self.device)
        qv = torch.as_tensor(q_val.astype(np.float32), device=self.device)
        qw = saturate(torch.relu(qv - tau_q), saturation)
        qw = qw * (torch.arange(qi.shape[1], device=self.device) < k_q)
        lens_all = (self.ptr[1:] - self.ptr[:-1])
        for s in range(0, nq, qchunk):
            qi_c, qw_c = qi[s:s + qchunk], qw[s:s + qchunk]
            m = qw_c > 0
            rows, cols = torch.nonzero(m, as_tuple=True)
            ent, w = qi_c[rows, cols], qw_c[rows, cols]
            scores = torch.zeros(qi_c.shape[0] * self.n_docs, device=self.device)
            lens = lens_all[ent]
            # split the triples so each pass stays inside the memory budget
            cum = torch.cumsum(lens, 0)
            splits, a = [0], 0
            while a < len(lens):
                base = cum[a - 1] if a > 0 else 0
                b = int(torch.searchsorted(cum, base + budget, right=True).item())
                a = max(b, a + 1)
                splits.append(min(a, len(lens)))
            for a, b in zip(splits[:-1], splits[1:]):
                e_, w_, r_ = ent[a:b], w[a:b], rows[a:b]
                l_ = lens[a:b]
                if int(l_.sum()) == 0:
                    continue
                start = self.ptr[e_]
                pos = torch.repeat_interleave(start, l_) + _within(l_)
                doc = self.doc[pos].long()
                pv = self.p[pos]
                rk = self.rank[pos]
                sd = saturate(torch.relu(pv - tau_d), saturation) * (rk < k_d)
                contrib = sd * torch.repeat_interleave(w_, l_)
                flat = torch.repeat_interleave(r_, l_) * self.n_docs + doc
                nz = contrib > 0
                scores.index_add_(0, flat[nz], contrib[nz])
            S = scores.view(qi_c.shape[0], self.n_docs)
            v, i = torch.topk(S, min(k, self.n_docs), dim=1)
            out_i[s:s + qi_c.shape[0], : i.shape[1]] = i.cpu().numpy()
            out_s[s:s + qi_c.shape[0], : v.shape[1]] = v.cpu().numpy()
            del scores, S
        return out_i, out_s


def _within(lens: torch.Tensor) -> torch.Tensor:
    """[0,1,..,l0-1, 0,1,..,l1-1, ...] for a vector of segment lengths."""
    starts = torch.cumsum(lens, 0) - lens
    total = int(lens.sum())
    return (torch.arange(total, device=lens.device)
            - torch.repeat_interleave(starts, lens))


def nnz_stats(val: np.ndarray, tau: float):
    m = val > tau
    n = m.sum(1)
    return dict(mean=float(n.mean()), median=float(np.median(n)),
                censored=float((n >= val.shape[1]).mean()))
