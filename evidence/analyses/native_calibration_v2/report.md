# Supplementary native matched-calibration results

This is an additional comparison of archived native baseline implementations. It is not a recalibration or reproduction of the separate-host TEDM/Timer-XL main table. All rows use fold 1 and seeds 1, 7, 19, 37, 73. Values are mean ± sample SD across training seeds; these SDs do not quantify market-path sampling uncertainty.

Each historical selected checkpoint is frozen. The same positive mean scalar in [0.05,1] is fitted by validation ES using 64 fixed residual-bootstrap scenarios; all point forecasts and residual distributions stay otherwise fixed. Test ES uses the original 256 residual-bootstrap draws and all-pairs ES, with the invariant pairwise term reused. TREFIN retains its original asymmetric Student-t scenarios. Hence this controls access to mean shrinkage, not the probabilistic adapter family.

| Model | MSE before x10^4 | MSE after x10^4 | ES before x10^2 | ES after x10^2 | Mean rho |
|---|---:|---:|---:|---:|---:|
| DLinear | 6.8994 ± 0.0049 | 6.8347 ± 0.0002 | 9.2726 ± 0.0010 | 9.2137 ± 0.0005 | 0.0500 |
| GRU | 6.8410 ± 0.0025 | 6.8347 ± 0.0013 | 9.2152 ± 0.0016 | 9.2131 ± 0.0008 | 0.3673 |
| HistGradientBoosting | 6.9005 ± 0.0133 | 6.8383 ± 0.0023 | 9.2642 ± 0.0144 | 9.2333 ± 0.0009 | 0.1458 |
| MASTER | 6.8309 ± 0.0080 | 6.8260 ± 0.0069 | 9.2096 ± 0.0043 | 9.2077 ± 0.0033 | 0.3318 |
| PatchTST | 6.8930 ± 0.0167 | 6.8337 ± 0.0007 | 9.2849 ± 0.0154 | 9.2148 ± 0.0011 | 0.0500 |
| RawHistoryMLP | 6.8981 ± 0.0101 | 6.8355 ± 0.0007 | 9.2895 ± 0.0063 | 9.2194 ± 0.0009 | 0.0500 |
| ShrunkVAR | 7.5290 ± 0.0000 | 6.8411 ± 0.0000 | 9.8157 ± 0.0000 | 9.2412 ± 0.0000 | 0.0500 |
| StateMLP | 6.8466 ± 0.0070 | 6.8311 ± 0.0042 | 9.2238 ± 0.0073 | 9.2123 ± 0.0022 | 0.4302 |
| iTransformer | 6.9261 ± 0.0151 | 6.8357 ± 0.0007 | 9.2939 ± 0.0088 | 9.2149 ± 0.0004 | 0.0500 |
| TREFIN (archived model) | 6.8542 ± 0.0414 | 6.8171 ± 0.0066 | 9.0060 ± 0.0359 | 8.9744 ± 0.0160 | archived |

## Paired temporal MSE comparisons

Five seed-specific losses are averaged at each origin before inference. Positive differences favor TREFIN. Primary intervals use circular blocks of 20 origins, 5,000 resamples; five-origin blocks are retained as a sensitivity check. The hypotheses and intervals concern this observed fold and fixed set of trained models.

| Calibrated native comparator | Difference x10^4 | 95% interval x10^4 | Holm-adjusted bootstrap p |
|---|---:|---:|---:|
| DLinear | 0.01754 | [-0.01160, 0.05097] | 1.0000 |
| GRU | 0.01760 | [-0.00618, 0.04670] | 1.0000 |
| HistGradientBoosting | 0.02120 | [-0.00389, 0.05205] | 0.8192 |
| MASTER | 0.00888 | [-0.01466, 0.03549] | 1.0000 |
| PatchTST | 0.01656 | [-0.01344, 0.05007] | 1.0000 |
| RawHistoryMLP | 0.01836 | [-0.01302, 0.05416] | 1.0000 |
| ShrunkVAR | 0.02400 | [-0.00082, 0.05327] | 0.5220 |
| StateMLP | 0.01399 | [-0.00417, 0.03516] | 1.0000 |
| iTransformer | 0.01862 | [-0.01120, 0.05319] | 1.0000 |

## Verification

All 45 runs passed. Independent exact all-pairs ES recomputation on two origins per run differed by at most 3.85e-08. Maximum CPU-versus-archived test-mean drift was 8.77e-06 return units; archived test means and training residuals were retained exactly. Validation-only selection used recovered CPU checkpoint inference, with maximum validation ES replay difference 7.62e-08. No training, checkpoint reselection, or test-driven hyperparameter selection was performed.
