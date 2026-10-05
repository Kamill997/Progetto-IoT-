"""
bekk_model.py
-------------
Maximum-likelihood estimation of the bivariate asymmetric BEKK(1,1) model used
in the paper (Section 3.2, equations 3-5) as "MGARCH-GJR-BEKK(C,D)", plus the
Wald tests of Table 6 for volatility spillover between the electricity and
cryptocurrency markets.

Model
-----
    a_t = H_t^(1/2) z_t,   z_t ~ N(0, I_2)
    H_t = C'C + X' a_{t-1} a_{t-1}' X + Y' H_{t-1} Y + Z' eta_{t-1} eta_{t-1}' Z
    eta_{t-1} = min(a_{t-1}, 0)              (element-wise)

C is upper triangular (3 free parameters); X, Y, Z are free 2x2 matrices (4
parameters each) -> 15 parameters total, estimated by maximizing the Gaussian
quasi log-likelihood:

    L = -0.5 * sum_t [ 2*log(2*pi) + log|H_t| + a_t' H_t^{-1} a_t ]

Hypotheses tested (paper, Section 3.2, and Table 6)
----------------------------------------------------
    x21 = y21 = 0  -> no volatility spillover FROM electricity demand TO crypto
    x12 = y12 = 0  -> no volatility spillover FROM crypto TO electricity demand
    z12 = z21 = 0  -> no asymmetric ("bad news") spillover between the markets

Each is tested with a Wald statistic theta_hat' * Cov(theta_hat)^{-1} * theta_hat,
which is chi-square distributed under the null, exactly as described in the
paper ("These hypotheses are tested using the Wald statistic which is
chi-square distributed").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import optimize, stats

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - graceful fallback if numba isn't installed
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def _decorator(f):
            return f
        return _decorator(args[0]) if args and callable(args[0]) else _decorator


PARAM_NAMES = [
    "c11", "c12", "c22",
    "x11", "x12", "x21", "x22",
    "y11", "y12", "y21", "y22",
    "z11", "z12", "z21", "z22",
]

_LOG_2PI = float(np.log(2 * np.pi))


def _unpack(theta: np.ndarray):
    c11, c12, c22, x11, x12, x21, x22, y11, y12, y21, y22, z11, z12, z21, z22 = theta
    C = np.array([[c11, c12], [0.0, c22]])
    X = np.array([[x11, x12], [x21, x22]])
    Y = np.array([[y11, y12], [y21, y22]])
    Z = np.array([[z11, z12], [z21, z22]])
    return C, X, Y, Z


def _pack(C: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return np.array([
        C[0, 0], C[0, 1], C[1, 1],
        X[0, 0], X[0, 1], X[1, 0], X[1, 1],
        Y[0, 0], Y[0, 1], Y[1, 0], Y[1, 1],
        Z[0, 0], Z[0, 1], Z[1, 0], Z[1, 1],
    ])


@njit(cache=True)
def _nll_core(theta: np.ndarray, a: np.ndarray) -> float:
    """Scalar-unrolled (2x2, hand-expanded) implementation of the negative
    Gaussian quasi log-likelihood of the GJR-BEKK(1,1) recursion. Because H_t
    is guaranteed positive semi-definite by construction (it is a sum of
    quadratic forms M'AM with A either C'C, an outer product, or the previous
    H_t, all PSD), only a tiny numerical floor is needed for stability. This
    is JIT-compiled with numba (falls back to plain Python if numba is not
    installed) since the recursive dependence on H_{t-1} prevents vectorizing
    the loop itself, and the 15-parameter MLE needs many thousands of
    likelihood evaluations during optimization and standard-error computation.
    """
    c11, c12, c22 = theta[0], theta[1], theta[2]
    x11, x12, x21, x22 = theta[3], theta[4], theta[5], theta[6]
    y11, y12, y21, y22 = theta[7], theta[8], theta[9], theta[10]
    z11, z12, z21, z22 = theta[11], theta[12], theta[13], theta[14]

    n = a.shape[0]

    h11c = c11 * c11
    h12c = c11 * c12
    h22c = c12 * c12 + c22 * c22

    h11 = h11c + 1e-6
    h12 = h12c
    h22 = h22c + 1e-6

    ll = 0.0
    for t in range(1, n):
        a1p = a[t - 1, 0]
        a2p = a[t - 1, 1]
        eta1 = a1p if a1p < 0.0 else 0.0
        eta2 = a2p if a2p < 0.0 else 0.0

        # X' * outer(a_prev, a_prev) * X
        m11 = a1p * a1p
        m12 = a1p * a2p
        m22 = a2p * a2p
        p11 = x11 * m11 + x21 * m12
        p12 = x11 * m12 + x21 * m22
        p21 = x12 * m11 + x22 * m12
        p22 = x12 * m12 + x22 * m22
        xt11 = p11 * x11 + p12 * x21
        xt12 = p11 * x12 + p12 * x22
        xt22 = p21 * x12 + p22 * x22

        # Y' * H_{t-1} * Y
        q11 = y11 * h11 + y21 * h12
        q12 = y11 * h12 + y21 * h22
        q21 = y12 * h11 + y22 * h12
        q22 = y12 * h12 + y22 * h22
        yt11 = q11 * y11 + q12 * y21
        yt12 = q11 * y12 + q12 * y22
        yt22 = q21 * y12 + q22 * y22

        # Z' * outer(eta_prev, eta_prev) * Z
        e11 = eta1 * eta1
        e12 = eta1 * eta2
        e22 = eta2 * eta2
        r11 = z11 * e11 + z21 * e12
        r12 = z11 * e12 + z21 * e22
        r21 = z12 * e11 + z22 * e12
        r22 = z12 * e12 + z22 * e22
        zt11 = r11 * z11 + r12 * z21
        zt12 = r11 * z12 + r12 * z22
        zt22 = r21 * z12 + r22 * z22

        h11 = h11c + xt11 + yt11 + zt11 + 1e-10
        h12 = h12c + xt12 + yt12 + zt12
        h22 = h22c + xt22 + yt22 + zt22 + 1e-10

        det = h11 * h22 - h12 * h12
        if det <= 1e-12:
            return 1e10

        a1, a2 = a[t, 0], a[t, 1]
        hinv11 = h22 / det
        hinv12 = -h12 / det
        hinv22 = h11 / det
        quad = a1 * a1 * hinv11 + 2.0 * a1 * a2 * hinv12 + a2 * a2 * hinv22
        logdet = np.log(det)

        ll += -0.5 * (2.0 * _LOG_2PI + logdet + quad)

    return -ll


def _neg_log_likelihood(theta: np.ndarray, a: np.ndarray) -> float:
    val = _nll_core(theta, a)
    if not np.isfinite(val):
        return 1e10
    return float(val)


@dataclass
class BEKKResult:
    theta: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    loglik: float
    converged: bool

    def param(self, name: str) -> float:
        return float(self.theta[PARAM_NAMES.index(name)])

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "param": PARAM_NAMES,
            "estimate": self.theta,
            "std_err": self.se,
        })


def _initial_guess(a: np.ndarray) -> np.ndarray:
    cov0 = np.cov(a.T)
    C0 = np.linalg.cholesky(cov0 * 0.3 + np.eye(2) * 1e-4).T
    X0 = np.array([[0.2, 0.02], [0.02, 0.2]])
    Y0 = np.array([[0.85, 0.02], [0.02, 0.85]])
    Z0 = np.array([[0.08, 0.01], [0.01, 0.08]])
    return _pack(C0, X0, Y0, Z0)


def _numerical_hessian(f, theta: np.ndarray, h: float = 1e-4) -> np.ndarray:
    n = len(theta)
    H = np.zeros((n, n))
    f0 = f(theta)
    for i in range(n):
        for j in range(i, n):
            t1 = theta.copy(); t1[i] += h; t1[j] += h
            t2 = theta.copy(); t2[i] += h; t2[j] -= h
            t3 = theta.copy(); t3[i] -= h; t3[j] += h
            t4 = theta.copy(); t4[i] -= h; t4[j] -= h
            val = (f(t1) - f(t2) - f(t3) + f(t4)) / (4 * h * h)
            H[i, j] = H[j, i] = val
    return H


# BEKK models are only identified up to simultaneous sign flips of a whole
# row/column of C, X, Y or Z (since only the quadratic forms M'.M enter the
# likelihood). The standard fix is to pin the sign of the diagonal ("own
# effect") entries to be non-negative, which removes the flat/ridge directions
# that otherwise stall the optimizer; the spillover (off-diagonal) entries are
# left free-signed since their sign is economically meaningful.
_BOUNDS = {
    "c11": (1e-4, 2.0), "c12": (-2.0, 2.0), "c22": (1e-4, 2.0),
    "x11": (0.0, 2.0), "x12": (-2.0, 2.0), "x21": (-2.0, 2.0), "x22": (0.0, 2.0),
    "y11": (0.0, 1.2), "y12": (-1.2, 1.2), "y21": (-1.2, 1.2), "y22": (0.0, 1.2),
    "z11": (0.0, 2.0), "z12": (-2.0, 2.0), "z21": (-2.0, 2.0), "z22": (0.0, 2.0),
}


def fit_gjr_bekk(a: np.ndarray, maxiter: int = 2000, n_starts: int = 4, seed: int = 0) -> BEKKResult:
    """Estimate the asymmetric BEKK(1,1) model on a demeaned bivariate residual
    series `a` (n_obs x 2), e.g. the residuals of a fitted VAR(p) mean equation.

    Uses a short multi-start (n_starts random perturbations of the analytic
    initial guess) since each likelihood evaluation is JIT-compiled and cheap
    (see _nll_core), so a handful of restarts costs well under a second and
    meaningfully reduces the chance of stopping at a poor local optimum.
    """
    rng = np.random.default_rng(seed)
    bounds = [_BOUNDS[name] for name in PARAM_NAMES]
    base_theta0 = _initial_guess(a)

    best_res = None
    for start in range(n_starts):
        if start == 0:
            theta0 = base_theta0
        else:
            noise = rng.normal(scale=0.05, size=len(base_theta0))
            theta0 = np.clip(base_theta0 + noise, [b[0] for b in bounds], [b[1] for b in bounds])

        res = optimize.minimize(
            _neg_log_likelihood,
            theta0,
            args=(a,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9},
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    # standard errors from the numerical Hessian of the negative log-likelihood
    try:
        hess = _numerical_hessian(lambda th: _neg_log_likelihood(th, a), best_res.x)
        cov = np.linalg.pinv(hess)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        cov = np.full((len(base_theta0), len(base_theta0)), np.nan)
        se = np.full(len(base_theta0), np.nan)

    return BEKKResult(theta=best_res.x, se=se, cov=cov, loglik=-best_res.fun, converged=best_res.success)


def wald_test(result: BEKKResult, param_names: list[str]) -> tuple[float, float, int]:
    """Joint Wald test that the listed parameters are simultaneously zero:
        W = theta_sub' * Cov(theta_sub)^{-1} * theta_sub  ~ chi2(len(param_names))
    Matches the paper's Wald statistic in Table 6 (e.g. x21 = y21 = 0).
    """
    idx = [PARAM_NAMES.index(p) for p in param_names]
    theta_sub = result.theta[idx]
    cov_sub = result.cov[np.ix_(idx, idx)]
    try:
        cov_inv = np.linalg.inv(cov_sub)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_sub)
    stat = float(theta_sub @ cov_inv @ theta_sub)
    df = len(param_names)
    p_value = float(1 - stats.chi2.cdf(stat, df))
    return stat, p_value, df


def spillover_wald_tests(result: BEKKResult) -> pd.DataFrame:
    """Reproduces the three Wald hypotheses of the paper's Table 6."""
    tests = [
        ("Electricity has no volatility spillover on Crypto (x21=y21=0)", ["x21", "y21"]),
        ("Crypto has no volatility spillover on Electricity (x12=y12=0)", ["x12", "y12"]),
        ("No asymmetric spillover between markets (z12=z21=0)", ["z12", "z21"]),
    ]
    rows = []
    for label, params in tests:
        stat, p_value, df = wald_test(result, params)
        rows.append({
            "Null Hypothesis": label,
            "Wald Statistic": round(stat, 3),
            "df": df,
            "P-Value": round(p_value, 4),
            "Conclusion": "Reject" if p_value < 0.05 else "Fail to Reject",
        })
    return pd.DataFrame(rows)
