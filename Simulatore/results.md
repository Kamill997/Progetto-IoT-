# Experimental report: VAR / Granger causality / GJR-BEKK

Data source: simulated (dgp.py, seed=7, n_obs=1500)

## Ground truth built into the simulated data

**VAR (mean) causal structure**
- electricity_causes_crypto (phi_12): `False`
- crypto_causes_electricity (phi_21): `False`

**BEKK volatility spillover structure**
- electricity_vol_spills_to_crypto (x21 or y21): `True`
- crypto_vol_spills_to_electricity (x12 or y12): `False`
- asymmetric_spillover (z12 or z21): `True`

## Table 1 - Descriptive statistics (paper Table 1 style)

| Series             |   Obs. |    Mean |   Variance |   Skewness |   Kurtosis |   Jarque-Bera |   LB-Q(10) | LB-Q2(10)   | ADF@I(0)   |
|:-------------------|-------:|--------:|-----------:|-----------:|-----------:|--------------:|-----------:|:------------|:-----------|
| crypto_return      |   1500 |  0.0057 |   0.015884 |     0.0381 |     3.2161 |          3.28 |       8.08 | 91.46***    | -36.432*** |
| electricity_demand |   1500 | -0.0005 |   0.001946 |     0.0308 |     2.8761 |          1.2  |      14.68 | 8.22        | -37.085*** |

## Table 2 - Stationarity (ADF) tests (paper Table 2 style)

| Series             | I(0) stat   | I(1) stat   | Stationary at   |
|:-------------------|:------------|:------------|:----------------|
| crypto_return      | -36.432***  | -15.869***  | I(0)            |
| electricity_demand | -37.085***  | -13.919***  | I(0)            |

## Tables 3/5 - Granger causality, crypto return <-> electricity demand (paper Table 3-5 style)

| Method        | Null Hypothesis                                         |   Statistic |   P-Value | Conclusion     |
|:--------------|:--------------------------------------------------------|------------:|----------:|:---------------|
| Parametric    | crypto_return does not Granger cause electricity_demand |       3.6   |    0.058  | Fail to Reject |
| Nonparametric | crypto_return does not Granger cause electricity_demand |      -0.154 |    0.915  | Fail to Reject |
| Parametric    | electricity_demand does not Granger cause crypto_return |       2.367 |    0.1241 | Fail to Reject |
| Nonparametric | electricity_demand does not Granger cause crypto_return |      -1.491 |    0.125  | Fail to Reject |

## Table 6 - GJR-BEKK volatility spillover Wald tests (paper Table 6 style)

VAR-BEKK log-likelihood: 3784.63 | optimizer converged: True

| Null Hypothesis                                               |   Wald Statistic |   df |   P-Value | Conclusion     |
|:--------------------------------------------------------------|-----------------:|-----:|----------:|:---------------|
| Electricity has no volatility spillover on Crypto (x21=y21=0) |            4.733 |    2 |    0.0938 | Fail to Reject |
| Crypto has no volatility spillover on Electricity (x12=y12=0) |            2.036 |    2 |    0.3613 | Fail to Reject |
| No asymmetric spillover between markets (z12=z21=0)           |            7.973 |    2 |    0.0186 | Reject         |

### Estimated BEKK parameters

| param   |   estimate |   std_err |
|:--------|-----------:|----------:|
| c11     |     0.0292 |    0.0104 |
| c12     |    -0.0011 |    0.0136 |
| c22     |     0.0304 |    0.0065 |
| x11     |     0.2789 |    0.0487 |
| x12     |    -0.0154 |    0.0245 |
| x21     |     0.125  |    0.1253 |
| x22     |     0.1842 |    0.0754 |
| y11     |     0.8347 |    0.0695 |
| y12     |     0.0543 |    0.0381 |
| y21     |     0.4392 |    0.334  |
| y22     |     0.5824 |    0.224  |
| z11     |     0      |    0.116  |
| z12     |    -0.0933 |    0.0343 |
| z21     |     0.3261 |    0.2243 |
| z22     |     0.3521 |    0.0996 |
