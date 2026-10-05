# Experimental report: VAR / Granger causality / GJR-BEKK

Data source: simulated (dgp.py, seed=42, n_obs=1200)

## Ground truth built into the simulated data

**VAR (mean) causal structure**
- electricity_causes_crypto (phi_12): `False`
- crypto_causes_electricity (phi_21): `False`

**BEKK volatility spillover structure**
- electricity_vol_spills_to_crypto (x21 or y21): `True`
- crypto_vol_spills_to_electricity (x12 or y12): `False`
- asymmetric_spillover (z12 or z21): `True`

## Table 1 - Descriptive statistics (paper Table 1 style)

| Series             |   Obs. |   Mean |   Variance |   Skewness |   Kurtosis |   Jarque-Bera | LB-Q(10)   | LB-Q2(10)   | ADF@I(0)   |
|:-------------------|-------:|-------:|-----------:|-----------:|-----------:|--------------:|:-----------|:------------|:-----------|
| crypto_return      |   1200 | -0.003 |   0.016003 |     0.0521 |     3.1869 |          2.29 | 25.74***   | 134.17***   | -12.891*** |
| electricity_demand |   1200 | -0.001 |   0.001882 |    -0.0956 |     3.0179 |          1.84 | 16.46*     | 11.05       | -31.927*** |

## Table 2 - Stationarity (ADF) tests (paper Table 2 style)

| Series             | I(0) stat   | I(1) stat   | Stationary at   |
|:-------------------|:------------|:------------|:----------------|
| crypto_return      | -12.891***  | -12.117***  | I(0)            |
| electricity_demand | -31.927***  | -11.629***  | I(0)            |

## Tables 3/5 - Granger causality, crypto return <-> electricity demand (paper Table 3-5 style)

| Method        | Null Hypothesis                                         |   Statistic |   P-Value | Conclusion     |
|:--------------|:--------------------------------------------------------|------------:|----------:|:---------------|
| Parametric    | crypto_return does not Granger cause electricity_demand |       0.665 |    0.4148 | Fail to Reject |
| Nonparametric | crypto_return does not Granger cause electricity_demand |      -0.463 |    0.63   | Fail to Reject |
| Parametric    | electricity_demand does not Granger cause crypto_return |       0.777 |    0.3782 | Fail to Reject |
| Nonparametric | electricity_demand does not Granger cause crypto_return |      -0.259 |    0.745  | Fail to Reject |

## Table 6 - GJR-BEKK volatility spillover Wald tests (paper Table 6 style)

VAR-BEKK log-likelihood: 3033.95 | optimizer converged: False

| Null Hypothesis                                               |   Wald Statistic |   df |   P-Value | Conclusion     |
|:--------------------------------------------------------------|-----------------:|-----:|----------:|:---------------|
| Electricity has no volatility spillover on Crypto (x21=y21=0) |            1.885 |    2 |    0.3897 | Fail to Reject |
| Crypto has no volatility spillover on Electricity (x12=y12=0) |            2.482 |    2 |    0.2891 | Fail to Reject |
| No asymmetric spillover between markets (z12=z21=0)           |            2.17  |    2 |    0.3379 | Fail to Reject |

### Estimated BEKK parameters

| param   |   estimate |   std_err |
|:--------|-----------:|----------:|
| c11     |     0.0273 |    0.0092 |
| c12     |    -0.0073 |    0.0151 |
| c22     |     0.0273 |    0.0068 |
| x11     |     0.2542 |    0.0545 |
| x12     |    -0.0055 |    0.0232 |
| x21     |    -0.0576 |    0.1662 |
| x22     |     0.2965 |    0.067  |
| y11     |     0.8397 |    0.096  |
| y12     |     0.0514 |    0.0341 |
| y21     |     0.542  |    0.4416 |
| y22     |     0.6068 |    0.1637 |
| z11     |     0.0911 |    0.1835 |
| z12     |    -0.065  |    0.0553 |
| z21     |    -0.0966 |    0.3308 |
| z22     |     0.0968 |    0.2357 |
