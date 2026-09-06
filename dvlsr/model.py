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
    """kind="mlp": the §D.1 residual MLP.  kind="linear": a residual *linear* map
    (zero-initialised, so also the identity at step 0) -- the capacity control for
    the follow-up study: a linear head cannot suppress individual entry directions
    the way an MLP can, so it tests whether co-adaptation to seen entries is what
    breaks held-out calibration."""

    def __init__(self, dim=768, hidden=768, t_init=20.0, b_init=0.0, kind="mlp"):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.lin1 = nn.Linear(dim, dim)
            nn.init.zeros_(self.lin1.weight)
            nn.init.zeros_(self.lin1.bias)
        else:
            self.lin1 = nn.Linear(dim, hidden)
            self.lin2 = nn.Linear(hidden, dim)
            nn.init.zeros_(self.lin2.weight)
            nn.init.zeros_(self.lin2.bias)
        self.t = nn.Parameter(torch.tensor(float(t_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))

    def forward(self, h):                       # h: (n, d) whitened token states
        if self.kind == "linear":
            return h + self.lin1(h)
        return h + self.lin2(F.gelu(self.lin1(h)))

    def logits(self, h, E, AB=None):            # E: (|V|, d) normalised entries
        """AB, when given, is a (2, |E|) per-entry affine map on the *cosine*, applied
        before the global scale and threshold:

            a_ij  = cos(g(h_i), v_j)
            a'_ij = A_j a_ij + B_j          background-normalised (dvlsr entry-norm)
            z_ij  = t a'_ij - b

        A_j and B_j are computed from the entry's own background statistics, not learned,
        so nothing here is a parameter indexed by j and insertion stays training-free."""
        gh = F.normalize(self(h), dim=-1)
        a = gh @ E.T
        if AB is not None:
            # the (units x entries) block is the memory budget of the whole step, so apply
            # the map without materialising a second copy of it
            a = (torch.addcmul(AB[1], a, AB[0]) if torch.is_grad_enabled()
                 else a.mul_(AB[0]).add_(AB[1]))
        return self.t * a - self.b


class EntryHead(nn.Module):
    """A residual map applied to *every* entry vector, mirroring the token-side head.

    §D.1 gives the token side a learned map and leaves the entry side frozen. That
    asymmetry has no principled justification, and the parameterized-vocabulary control
    shows it costs ~33% MRR@10: with L2-normalised rows the whole benefit of a learned
    vocabulary comes from moving entry *directions*.

    This is one shared function, not a parameter per entry, so it satisfies the hard
    constraint and it applies to a newly inserted entry exactly as to a trained one:
    encode the text, whiten, apply this map. What it can capture is the *systematic*
    part of the correction; per-entry idiosyncrasy remains out of reach by construction,
    which is precisely the decomposition the experiment is meant to measure.
    """

    def __init__(self, dim=768, hidden=768):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden)
        self.lin2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, E):
        return F.normalize(E + self.lin2(F.gelu(self.lin1(E))), dim=-1)


def segment_max(z, row, n_rows):
    """max over the units of each text, differentiable (autograd handles amax)."""
    out = z.new_full((n_rows, z.shape[1]), -1e4)
    return out.scatter_reduce(0, row.unsqueeze(1).expand_as(z), z, reduce="amax",
                              include_self=True)


def sparse_rep(head, h, row, n_rows, E, mask=None, AB=None):
    """s_j(x) = log(1 + ReLU(max_i z_ij)), optionally over a masked entry subset."""
    z = head.logits(h, E, AB)
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


def kmeans_labels(X: torch.Tensor, k: int, seed: int = 0, iters: int = 30) -> torch.Tensor:
    """Spherical k-means over L2-normalised rows (the whitened entry space).

    Used for cluster-wise vocabulary dropout and for the meta-held-out clusters, so
    the regions dropped in training are of the same kind as the §D.4 cluster split.
    """
    g = torch.Generator(device=X.device)
    g.manual_seed(int(seed))
    n = X.shape[0]
    C = X[torch.randperm(n, generator=g, device=X.device)[:k]].clone()
    lab = None
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        S = torch.zeros_like(C).index_add_(0, lab, X)
        cnt = torch.bincount(lab, minlength=k).unsqueeze(1).to(X.dtype)
        empty = (cnt.squeeze(1) == 0)
        C = F.normalize(S / cnt.clamp(min=1), dim=-1)
        if empty.any():                      # re-seed empty clusters on random rows
            C[empty] = X[torch.randint(0, n, (int(empty.sum()),), generator=g,
                                       device=X.device)]
    return lab


def calibration_loss(S: torch.Tensor, meta: torch.Tensor, trained: torch.Tensor,
                     decile: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Meta-held-out calibration: per frequency decile, the squared log-ratio between
    the batch-mean activation of entries that are *never* in the ranking objective
    (meta) and of entries that are (trained). The trained side is detached, so the
    gradient only moves the never-trained region towards the trained one. This is the
    training-time analogue of Pilot D's signed gap_r.

    S: (n_texts, n_entries) sparse representation over the seen entries.
    """
    m = S.mean(0)
    loss = S.new_zeros(())
    n = 0
    for d in range(10):
        sel = decile == d
        a = m[sel & meta]
        b = m[sel & trained]
        if a.numel() == 0 or b.numel() == 0:
            continue
        loss = loss + (torch.log(a.mean() + eps) - torch.log(b.mean().detach() + eps)) ** 2
        n += 1
    return loss / max(n, 1)


def entry_norm_affine(mu, sd, mu_star, sd_star, eps=1e-6):
    """Per-entry affine that maps an entry's background similarity distribution onto the
    target (mu*, sd*) of comparable seen entries:  a' = A a + B  with

        A_j = sd*_j / sd_j          B_j = mu*_j - mu_j A_j

    For an entry whose background statistics already equal the target this is the identity,
    so the map is a correction of per-entry idiosyncrasy rather than a global rescaling."""
    A = sd_star / sd.clamp(min=eps)
    return torch.stack([A, mu_star - mu * A])
