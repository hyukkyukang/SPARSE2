"""Ranking metrics and the paired bootstrap of §0.7."""
from __future__ import annotations
import numpy as np


def rr_at_k(ranked: np.ndarray, rel: set, k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if int(d) in rel:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: np.ndarray, rel: set, k: int) -> float:
    if not rel:
        return 0.0
    return len(rel & set(int(x) for x in ranked[:k])) / len(rel)


def evaluate(run: dict[int, np.ndarray], qrels: dict[int, set],
             ks=(10, 100, 1000)) -> tuple[dict, dict]:
    """Returns (means, per-query arrays). Queries with no qrels are skipped."""
    qids = sorted(q for q in qrels if q in run)
    per = {"mrr@10": [], **{f"r@{k}": [] for k in ks}}
    for q in qids:
        r = run[q]
        per["mrr@10"].append(rr_at_k(r, qrels[q], 10))
        for k in ks:
            per[f"r@{k}"].append(recall_at_k(r, qrels[q], k))
    per = {k: np.asarray(v) for k, v in per.items()}
    return {k: float(v.mean()) for k, v in per.items()}, dict(per, qids=np.asarray(qids))


def bootstrap_ci(x: np.ndarray, n: int = 10_000, alpha: float = 0.05, seed: int = 0):
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000,
                     alpha: float = 0.05, seed: int = 0):
    """Paired bootstrap over queries for the difference a - b (§0.7)."""
    assert len(a) == len(b)
    g = np.random.default_rng(seed)
    d = a - b
    idx = g.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    lo, hi = float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))
    return dict(diff=float(d.mean()), lo=lo, hi=hi,
                p_two_sided=float(2 * min((m <= 0).mean(), (m >= 0).mean())),
                excludes_zero=bool(lo > 0 or hi < 0))


def ratio_bootstrap(num_a: np.ndarray, num_b: np.ndarray,
                    den_a: np.ndarray, den_b: np.ndarray,
                    n: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """CI for rho = mean(num_a - num_b) / mean(den_a - den_b), resampling queries jointly.

    Used for Pilot D's recovery ratio; the denominator is the oracle's own gap.
    """
    g = np.random.default_rng(seed)
    m = len(num_a)
    idx = g.integers(0, m, size=(n, m))
    num = (num_a - num_b)[idx].mean(1)
    den = (den_a - den_b)[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(den) > 1e-12, num / den, np.nan)
    r = r[np.isfinite(r)]
    point = float((num_a - num_b).mean() / (den_a - den_b).mean()) \
        if abs((den_a - den_b).mean()) > 1e-12 else float("nan")
    return dict(rho=point, lo=float(np.quantile(r, alpha / 2)),
                hi=float(np.quantile(r, 1 - alpha / 2)),
                den=float((den_a - den_b).mean()))
