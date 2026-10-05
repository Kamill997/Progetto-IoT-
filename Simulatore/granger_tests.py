"""
granger_tests.py
-----------------
Implements the two lead-lag testing procedures used in the paper (Section 3.2,
hypotheses H01-H04) to fill in Tables 3, 4 and 5:

1. parametric_granger_causality()
   Linear (parametric) Granger causality within a bivariate VAR(p), where p is
   chosen by AIC (as stated in the paper). Tests, via an F/Wald test on the
   relevant lag coefficients:
       H01: crypto does NOT Granger-cause electricity demand
       H02: electricity demand does NOT Granger-cause crypto
   Implemented with statsmodels (VAR order selection + grangercausalitytests).

2. nonparametric_granger_causality()
   A Diks & Panchenko (2006)-style nonparametric test, matching the paper's
   H03/H04 (equal conditional distributions rather than equal conditional
   means). It measures whether the local co-occurrence of (X_t, Y_t, X_{t+1})
   departs from what independence between Y_t and X_{t+1} given X_t would imply.

Implementation note on the nonparametric test
-----------------------------------------------
The original Diks-Panchenko (2006) paper derives a closed-form asymptotic
variance for the test statistic from a 6-dimensional U-statistic kernel, which
is intricate to reproduce exactly. Here the same local-density-ratio statistic
is computed, but its null distribution is obtained by a *stationary bootstrap*
(block-shuffling the driving series many times to destroy the X<->Y cross
dependence while preserving each series' own serial dependence) and the
p-value is read off the empirical null distribution. This is a standard,
transparent substitute for the closed-form variance formula and is exact up to
Monte-Carlo error in the number of bootstrap replications. The bandwidth and
embedding dimension default to the same values reported in the paper's table
notes (bandwidth = 0.5, embedding dimension = 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests


@dataclass
class GrangerResult:
    method: str
    null_hypothesis: str
    statistic: float
    p_value: float
    conclusion: str

    def stars(self) -> str:
        if self.p_value < 0.01:
            return "***"
        if self.p_value < 0.05:
            return "**"
        if self.p_value < 0.10:
            return "*"
        return ""


def _conclusion(p_value: float, alpha: float = 0.05) -> str:
    return "Reject" if p_value < alpha else "Fail to Reject"


def select_var_order(data: pd.DataFrame, maxlags: int = 10) -> int:
    """Select VAR(p) order by AIC, as stated in the paper (Section 3.2)."""
    model = VAR(data)
    sel = model.select_order(maxlags=maxlags)
    order = sel.aic
    return max(order, 1)


def parametric_granger_causality(
    data: pd.DataFrame,
    cause: str,
    effect: str,
    maxlags: int = 10,
) -> GrangerResult:
    """Linear/parametric bivariate Granger causality test: does `cause`
    Granger-cause `effect`? Mirrors paper hypotheses H01/H02 and Tables 3-5.
    """
    p = select_var_order(data[[effect, cause]], maxlags=maxlags)
    test = grangercausalitytests(data[[effect, cause]], maxlag=[p])
    f_stat, p_value, _, _ = test[p][0]["ssr_ftest"]

    null_hyp = f"{cause} does not Granger cause {effect}"
    return GrangerResult(
        method="Parametric",
        null_hypothesis=null_hyp,
        statistic=float(f_stat),
        p_value=float(p_value),
        conclusion=_conclusion(p_value),
    )


def _embed(series: np.ndarray, dim: int) -> np.ndarray:
    """Delay-vector embedding: row t = (series[t-dim+1], ..., series[t])."""
    n = len(series)
    return np.array([series[t - dim + 1 : t + 1] for t in range(dim - 1, n)])


def _pairwise_close(mat: np.ndarray, eps: float) -> np.ndarray:
    """Boolean matrix: True where two rows are within `eps` in max-norm."""
    diff = np.abs(mat[:, None, :] - mat[None, :, :]).max(axis=2)
    close = diff < eps
    np.fill_diagonal(close, False)
    return close


def _diks_panchenko_statistic(
    x: np.ndarray, y: np.ndarray, lag: int, embed_dim: int, bandwidth: float
) -> float:
    """Compute the local-density-ratio statistic used by nonparametric_granger_causality.

    Tests whether Y_t helps predict X_{t+1} beyond X_t's own lags, i.e. whether
    f(x_future | x_past, y_past) differs from f(x_future | x_past).
    """
    n = len(x) - max(embed_dim - 1, lag)
    x_hist = _embed(x, embed_dim)[:-lag] if lag > 0 else _embed(x, embed_dim)
    y_hist = _embed(y, embed_dim)[:-lag] if lag > 0 else _embed(y, embed_dim)
    x_future = x[embed_dim - 1 + lag :]

    m = min(len(x_hist), len(y_hist), len(x_future))
    x_hist, y_hist, x_future = x_hist[:m], y_hist[:m], x_future[:m].reshape(-1, 1)

    close_x = _pairwise_close(x_hist, bandwidth)
    close_y = _pairwise_close(y_hist, bandwidth)
    close_z = _pairwise_close(x_future, bandwidth)

    counts_x = close_x.sum(axis=1)
    counts_xy = (close_x & close_y).sum(axis=1)
    counts_xz = (close_x & close_z).sum(axis=1)
    counts_xyz = (close_x & close_y & close_z).sum(axis=1)

    valid = (counts_x > 0) & (counts_xy > 0)
    if valid.sum() == 0:
        return 0.0

    ratio = np.divide(
        counts_xyz[valid], counts_xy[valid], out=np.zeros(valid.sum()), where=counts_xy[valid] > 0
    ) - np.divide(
        counts_xz[valid], counts_x[valid], out=np.zeros(valid.sum()), where=counts_x[valid] > 0
    )
    return float(np.mean(ratio))


def nonparametric_granger_causality(
    data: pd.DataFrame,
    cause: str,
    effect: str,
    lag: int = 1,
    embed_dim: int = 2,
    bandwidth: float = 0.5,
    n_bootstrap: int = 199,
    seed: int = 0,
    standardize: bool = True,
) -> GrangerResult:
    """Nonparametric Granger causality test (Diks-Panchenko style), mirroring
    paper hypotheses H03/H04 and the "Nonparametric" rows of Tables 3-5.

    NOTE: this is O(n^2) in the sample size (all pairwise distances), so for
    long series `data` should be sub-sampled by the caller before calling this
    (run_experiment.py does this and documents the sub-sample size used).
    """
    rng = np.random.default_rng(seed)

    x = data[effect].to_numpy(dtype=float)
    y = data[cause].to_numpy(dtype=float)
    if standardize:
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()

    observed = _diks_panchenko_statistic(x, y, lag, embed_dim, bandwidth)

    # stationary bootstrap null distribution: shuffle y in contiguous blocks to
    # destroy cross-dependence with x while keeping y's own autocorrelation
    block = max(int(round(len(y) ** (1 / 3))), 2)
    null_stats = np.empty(n_bootstrap)
    n = len(y)
    for b in range(n_bootstrap):
        y_boot = _stationary_bootstrap_series(y, block, rng)
        null_stats[b] = _diks_panchenko_statistic(x, y_boot, lag, embed_dim, bandwidth)

    p_value = float((np.sum(np.abs(null_stats) >= abs(observed)) + 1) / (n_bootstrap + 1))
    # report a pseudo z-score against the bootstrap null for readability
    sd = null_stats.std() if null_stats.std() > 0 else np.nan
    z_stat = observed / sd if sd and not np.isnan(sd) else np.nan

    null_hyp = f"{cause} does not Granger cause {effect}"
    return GrangerResult(
        method="Nonparametric",
        null_hypothesis=null_hyp,
        statistic=float(z_stat) if not np.isnan(z_stat) else observed,
        p_value=p_value,
        conclusion=_conclusion(p_value),
    )


def _stationary_bootstrap_series(series: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(series)
    out = np.empty(n)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        length = min(block, n - i)
        idx = (start + np.arange(length)) % n
        out[i : i + length] = series[idx]
        i += length
    return out


def results_table(results: list[GrangerResult]) -> pd.DataFrame:
    rows = [
        {
            "Method": r.method,
            "Null Hypothesis": r.null_hypothesis,
            "Statistic": round(r.statistic, 3),
            "P-Value": round(r.p_value, 4),
            "Conclusion": r.conclusion,
        }
        for r in results
    ]
    return pd.DataFrame(rows)
