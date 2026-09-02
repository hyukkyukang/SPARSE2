"""§0.5 — ZCA whitening, estimated per space and frozen.

    mu = mean(X);  Sigma = cov(X);  Sigma += eps*I with eps = 0.01*tr(Sigma)/d
    W = Sigma^{-1/2}   (ZCA, symmetric)
    whiten(x) = (x - mu) @ W
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path

from . import paths


def estimate(X: np.ndarray | torch.Tensor, eps_scale: float = paths.WHITEN_EPS):
    X = torch.as_tensor(np.asarray(X)).float()
    mu = X.mean(0)
    Xc = X - mu
    d = X.shape[1]
    Sig = (Xc.T @ Xc) / (X.shape[0] - 1)
    eps = eps_scale * torch.diagonal(Sig).sum() / d
    Sig = Sig + eps * torch.eye(d)
    evals, evecs = torch.linalg.eigh(Sig.double())
    evals = evals.clamp(min=1e-12)
    W = (evecs @ torch.diag(evals.rsqrt()) @ evecs.T).float()
    return mu.numpy(), W.numpy(), float(eps)


def save(path: Path, mu, W, eps, meta=None):
    np.savez(path, mu=np.asarray(mu, np.float32), W=np.asarray(W, np.float32),
             eps=np.float32(eps), meta=np.asarray(str(meta or "")))


def load(path: Path):
    z = np.load(path, allow_pickle=False)
    return z["mu"], z["W"]


class Transform:
    """One of the three §0.5 conditions: raw / centered / whitened."""

    def __init__(self, mode: str, mu=None, W=None, device="cuda"):
        assert mode in ("raw", "centered", "whitened")
        self.mode = mode
        self.mu = None if mu is None else torch.as_tensor(mu, dtype=torch.float32, device=device)
        self.W = None if W is None else torch.as_tensor(W, dtype=torch.float32, device=device)

    def __call__(self, x: torch.Tensor, normalize=True) -> torch.Tensor:
        x = x.float()
        if self.mode != "raw":
            x = x - self.mu
        if self.mode == "whitened":
            x = x @ self.W
        if normalize:
            x = torch.nn.functional.normalize(x, dim=-1)
        return x


def cov_compare(mu_a, W_a_cov, mu_b, W_b_cov):
    """Relative mean shift and Frobenius covariance difference (§0.5 query/doc check)."""
    dmu = float(np.linalg.norm(mu_a - mu_b) / (np.linalg.norm(mu_b) + 1e-12))
    dcov = float(np.linalg.norm(W_a_cov - W_b_cov) / (np.linalg.norm(W_b_cov) + 1e-12))
    return dmu, dcov
