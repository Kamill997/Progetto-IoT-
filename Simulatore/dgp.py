"""
dgp.py
------
Simulates synthetic data from the same statistical model used in the paper

    Okorie, D.I., Gnatchiglo, J.M., Wesseh, P.K. (2024)
    "Electricity and cryptocurrency mining: An empirical contribution"
    Heliyon 10, e33483.

Reproduced equations
---------------------
Mean equation (paper eq. 1-2), bivariate VAR(p):

    A_t = mu_t + a_t
    mu_t = E(A_t | F_{t-1})
    phi_i(L, p_i) A_it = b_i0 + b_i1 * t + a_it

Conditional covariance, asymmetric BEKK (paper eq. 3-5), "MGARCH-GJR-BEKK(C,D)":

    a_t = H_t^(1/2) z_t ,          z_t ~ N(0, I)
    H_t = C'C + X' a_{t-1} a_{t-1}' X + Y' H_{t-1} Y + Z' eta_{t-1} eta_{t-1}' Z
    eta_{t-1} = min(a_{t-1}, 0)        (element-wise; captures the GJR asymmetric term)

A_t is a 2x1 vector: index 0 = cryptocurrency return, index 1 = electricity demand
(growth). C is upper-triangular (2x2) so that H_t is guaranteed positive
semi-definite by construction; X, Y, Z are free 2x2 matrices.

Why simulate instead of scraping the paper's original data
------------------------------------------------------------
The paper's series (Coin Market Cap prices, Quandl/Etherscan hash rates, World
Bank electricity demand, 2013-2019) are not shipped with the article and are not
reachable from this environment. Building a *simulator* of the model itself -
rather than trying to re-download the original data - is also more useful for a
methodology exercise: it lets you fix the "true" causal/spillover structure
(which off-diagonal parameters are non-zero) and then check whether the
estimators/tests in granger_tests.py and bekk_model.py recover it. This is the
standard Monte-Carlo way to validate an econometric procedure before trusting it
on real, noisy data (which run_experiment.py can also accept via --csv).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class DGPParams:
    """Parameters of the VAR(1)-GJR-BEKK(1,1) data-generating process.

    Naming mirrors the paper: index 0 = cryptocurrency return (A1t),
    index 1 = electricity demand (A2t).
    """

    # --- mean equation: A_t = intercept + Phi @ A_{t-1} + a_t ---
    intercept: np.ndarray = field(default_factory=lambda: np.array([0.0005, 0.0002]))
    Phi: np.ndarray = field(default_factory=lambda: np.array([[0.03, 0.00],
                                                                [0.00, 0.05]]))

    # --- BEKK constant term C (upper triangular) ---
    C: np.ndarray = field(default_factory=lambda: np.array([[0.03, 0.005],
                                                               [0.00, 0.02]]))

    # --- ARCH matrix X: short-term volatility spillover ---
    # x21 != 0  -> electricity-demand shocks feed Bitcoin's conditional variance
    # x12 != 0  -> Bitcoin shocks feed electricity demand's conditional variance
    X: np.ndarray = field(default_factory=lambda: np.array([[0.25, 0.00],
                                                               [0.18, 0.20]]))

    # --- GARCH matrix Y: long-term persistence / spillover ---
    Y: np.ndarray = field(default_factory=lambda: np.array([[0.90, 0.00],
                                                               [0.15, 0.85]]))

    # --- asymmetric (GJR) matrix Z: negative-shock spillover ---
    Z: np.ndarray = field(default_factory=lambda: np.array([[0.10, 0.00],
                                                               [0.05, 0.10]]))

    def var_causal_structure(self) -> dict:
        """Human-readable description of which Granger-causal links are
        actually present in Phi, so results can be checked against ground truth."""
        return {
            "electricity_causes_crypto (phi_12)": bool(abs(self.Phi[0, 1]) > 1e-8),
            "crypto_causes_electricity (phi_21)": bool(abs(self.Phi[1, 0]) > 1e-8),
        }

    def bekk_spillover_structure(self) -> dict:
        return {
            "electricity_vol_spills_to_crypto (x21 or y21)": bool(
                abs(self.X[1, 0]) > 1e-8 or abs(self.Y[1, 0]) > 1e-8
            ),
            "crypto_vol_spills_to_electricity (x12 or y12)": bool(
                abs(self.X
                [0, 1]) > 1e-8 or abs(self.Y[0, 1]) > 1e-8
            ),
            "asymmetric_spillover (z12 or z21)": bool(
                abs(self.Z[0, 1]) > 1e-8 or abs(self.Z[1, 0]) > 1e-8
            ),
        }


def simulate(
    params: DGPParams,
    n_obs: int = 1500,
    n_burn: int = 500,
    seed: int | None = 42,
):
    """Simulate one path of length n_obs from the VAR(1)-GJR-BEKK(1,1) model.

    Returns
    -------
    A : np.ndarray, shape (n_obs, 2)
        Simulated series [crypto_return, electricity_demand_growth].
    H : np.ndarray, shape (n_obs, 2, 2)
        The conditional covariance path actually used (useful for diagnostics /
        for checking an estimated BEKK model against the truth).
    """
    rng = np.random.default_rng(seed)

    n_total = n_obs + n_burn
    C, X, Y, Z, Phi, intercept = params.C, params.X, params.Y, params.Z, params.Phi, params.intercept

    A = np.zeros((n_total, 2))
    a = np.zeros((n_total, 2))
    H = np.zeros((n_total, 2, 2))

    # initialize H0 at the (approximate) unconditional covariance implied by C alone
    H[0] = C.T @ C + np.eye(2) * 1e-4
    A[0] = intercept

    for t in range(1, n_total):
        a_prev = a[t - 1]
        eta_prev = np.minimum(a_prev, 0.0)  # negative part only -> GJR asymmetry
        H_prev = H[t - 1]

        H_t = (
            C.T @ C
            + X.T @ np.outer(a_prev, a_prev) @ X
            + Y.T @ H_prev @ Y
            + Z.T @ np.outer(eta_prev, eta_prev) @ Z
        )
        # enforce numerical symmetry / mild regularization for stability
        H_t = 0.5 * (H_t + H_t.T) + np.eye(2) * 1e-10
        H[t] = H_t

        L = np.linalg.cholesky(H_t)
        z = rng.standard_normal(2)
        a_t = L @ z
        a[t] = a_t

        mu_t = intercept + Phi @ A[t - 1]
        A[t] = mu_t + a_t

    return A[n_burn:], H[n_burn:]


def to_dataframe(A: np.ndarray, freq_start: str = "2013-04-28"):
    """Wrap the simulated array in a pandas DataFrame with a daily date index,
    labeled the same way as the paper's two series."""
    import pandas as pd

    idx = pd.date_range(start=freq_start, periods=A.shape[0], freq="D")
    return pd.DataFrame(A, index=idx, columns=["crypto_return", "electricity_demand"])
