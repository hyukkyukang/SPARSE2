"""§0.6 — the training-free sparse encoding rule.

  a_ij = cos(h~_i, v~_j) ; p_j = max_i a_ij ; s_j = relu(p_j - tau) ; top-k truncation
  score(q,d) = sum_j s_j(q) s_j(d)

Profiles are streamed in chunks and never stored in full (§0.7).
"""
from __future__ import annotations
import numpy as np
import torch


def profile_maxpool(H: torch.Tensor, row: torch.Tensor, n_rows: int,
                    V: torch.Tensor, chunk_units: int = 8192,
                    chunk_entries: int = 30000) -> torch.Tensor:
    """max_i cos(h_i, v_j) grouped by row. H, V must already be whitened+normalised.

    V may be (|V|, d) or (|V|, m, d) for a multi-prototype representation.
    """
    out = torch.full((n_rows, V.shape[0]), -1.0, device=V.device, dtype=torch.float32)
    for s in range(0, H.shape[0], chunk_units):
        h = H[s:s + chunk_units]
        r = row[s:s + chunk_units]
        for e in range(0, V.shape[0], chunk_entries):
            Vc = V[e:e + chunk_entries]
            if Vc.dim() == 3:                 # multi-prototype: max over an entry's atoms
                n, m, d = Vc.shape
                a = (h @ Vc.reshape(-1, d).T).float().reshape(h.shape[0], n, m).amax(-1)
            else:
                a = (h @ Vc.T).float()
            out[:, e:e + chunk_entries].scatter_reduce_(
                0, r.unsqueeze(1).expand(-1, a.shape[1]), a, reduce="amax")
    return out


def sparsify(P: torch.Tensor, tau: float, topk: int, saturation: str = "none"):
    """ReLU(p - tau), then top-k truncation. Returns (idx, val) ragged as padded tensors."""
    S = torch.relu(P - tau)
    k = min(topk, S.shape[1])
    val, idx = torch.topk(S, k, dim=1)
    if saturation == "log1p":
        val = torch.log1p(val)
    return idx, val


def nnz_before_truncation(P: torch.Tensor, tau: float) -> torch.Tensor:
    return (P > tau).sum(1)


def find_tau(P: torch.Tensor, target_nnz: float, lo=-0.5, hi=1.0, iters=40) -> float:
    """Binary-search tau so mean nnz per text (before truncation) hits the target (§0.6)."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = nnz_before_truncation(P, mid).float().mean().item()
        if m > target_nnz:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def per_token_percentile(P_tokens: torch.Tensor, tau: float) -> float:
    """Diagnostic: what per-token percentile the chosen tau corresponds to (§0.6)."""
    return float((P_tokens < tau).float().mean().item() * 100)


def to_csr(idx: np.ndarray, val: np.ndarray, n_cols: int):
    """Ragged padded (idx,val) with val==0 padding -> scipy CSR."""
    import scipy.sparse as sp
    n = idx.shape[0]
    mask = val > 0
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(mask.sum(1))
    return sp.csr_matrix((val[mask].astype(np.float32), idx[mask].astype(np.int32), indptr),
                         shape=(n, n_cols))
