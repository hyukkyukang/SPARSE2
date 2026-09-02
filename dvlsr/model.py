"""Pilot D model (§D.1).

    h~   = W_H (h - mu_H)                     frozen whitening
    g(h~) = h~ + Linear2(GELU(Linear1(h~)))   residual MLP, Linear2 zero-init
    v~_j = normalise(W_V (v_j - mu_V))        frozen entry matrix
    z_ij = t * cos(g(h~_i), v~_j) - b         learnable scalars t, b
    s_j(x) = log(1 + ReLU(max_i z_ij))

Nothing in the model is indexed by j: the entry side is data, not parameters.
That is the hard constraint the whole study rests on.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, dim=768, hidden=768, t_init=20.0, b_init=0.0):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden)
        self.lin2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)
        self.t = nn.Parameter(torch.tensor(float(t_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))

    def forward(self, h):                       # h: (n, d) whitened token states
        return h + self.lin2(F.gelu(self.lin1(h)))

    def logits(self, h, E):                     # E: (|V|, d) normalised entries
        gh = F.normalize(self(h), dim=-1)
        return self.t * (gh @ E.T) - self.b


def segment_max(z, row, n_rows):
    """max over the units of each text, differentiable (autograd handles amax)."""
    out = z.new_full((n_rows, z.shape[1]), -1e4)
    return out.scatter_reduce(0, row.unsqueeze(1).expand_as(z), z, reduce="amax",
                              include_self=True)


def sparse_rep(head, h, row, n_rows, E, mask=None):
    """s_j(x) = log(1 + ReLU(max_i z_ij)), optionally over a masked entry subset."""
    z = head.logits(h, E)
    if mask is not None:
        z = z[:, mask]
    p = segment_max(z, row, n_rows)
    return torch.log1p(torch.relu(p))


def flops(s):
    """FLOPS regulariser: sum_j (batch-mean s_j)^2."""
    return (s.mean(0) ** 2).sum()


def info_nce(sq, sd, pos_idx, temperature=1.0):
    """InfoNCE over {positive, hard negatives, all other in-batch passages}."""
    scores = sq @ sd.T
    return F.cross_entropy(scores / temperature, pos_idx), scores


def build_entry_matrix(V_raw: np.ndarray, seen: np.ndarray, device="cuda",
                       eps_scale=0.01):
    """Entry side, with mu_V and W_V estimated over SEEN entries only (§D.1 leakage rule).

    Returns the transformed+normalised matrix for *all* entries: held-out rows are
    produced by the same frozen map, which is exactly what "insertion" means.
    """
    from .whitening import estimate
    mu, W, _ = estimate(V_raw[seen].astype(np.float32), eps_scale)
    X = torch.as_tensor(V_raw.astype(np.float32), device=device)
    X = (X - torch.as_tensor(mu, device=device)) @ torch.as_tensor(W, device=device)
    return F.normalize(X, dim=-1), mu, W
