# Matched structural retraining summary

Status: **complete**. Completed 25/25 requested runs.

All five variants are retained. Means and sample SD describe training-seed variation. Paired inference averages matched seed losses within each chronological origin; all assets and five leads remain grouped. Positive candidate-minus-Full loss differences favor Full. Primary circular block length is20;5 and10 are sensitivity checks. MSE and ES p values are Holm-adjusted over the four variants separately within each fold/block. MAE p values are unadjusted.

NoGraph removes the A input/rollout/decoder channel and graph-target loss. It retains the market state, including its residual-dependence summary; it does not remove every graph-derived quantity.

## Fold1 calibrated results

MSE is multiplied by10^4; all other displayed metrics are multiplied by10^2. RMSE is computed for each run before seed aggregation.

| Variant | Seeds | MSE | RMSE | MAE | ES | msIC | msIR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 5 | 6.8170 ± 0.0055 | 2.6109 ± 0.0010 | 1.6703 ± 0.0003 | 8.9754 ± 0.0191 | 0.6304 ± 0.1926 | 1.2790 ± 0.3972 |
| NoGraph | 5 | 6.8268 ± 0.0093 | 2.6128 ± 0.0018 | 1.6717 ± 0.0011 | 9.0569 ± 0.0296 | 0.7095 ± 0.4137 | 1.4313 ± 0.8275 |
| SingleSigned | 5 | 6.8170 ± 0.0055 | 2.6109 ± 0.0011 | 1.6703 ± 0.0003 | 8.9756 ± 0.0193 | 0.6354 ± 0.2102 | 1.2877 ± 0.4314 |
| SingleUnsigned | 5 | 6.8193 ± 0.0043 | 2.6114 ± 0.0008 | 1.6703 ± 0.0003 | 8.9495 ± 0.0211 | 0.6598 ± 0.2770 | 1.3398 ± 0.5649 |
| NoStateSupervision | 5 | 6.8192 ± 0.0044 | 2.6114 ± 0.0008 | 1.6701 ± 0.0003 | 8.9560 ± 0.0183 | 0.3492 ± 0.2236 | 0.7160 ± 0.4519 |

Primary paired comparisons (raw loss units; no benefit claim is inferred from ranking alone):

| Candidate − Full | Metric | Matched seeds | Mean difference |95% CI|Raw p|Holm p|
|---|---|---:|---:|---|---:|---:|
|NoGraph|mse|5|9.8368e-07|[-7.22033e-07,2.96553e-06]|0.2975|1.0000|
|NoGraph|energy_score|5|0.000815215|[0.000405052,0.00138668]|0.0054|0.0108|
|SingleSigned|mse|5|1.81306e-09|[-6.99623e-08,7.5519e-08]|0.9598|1.0000|
|SingleSigned|energy_score|5|2.70449e-06|[-4.21809e-06,1.1243e-05]|0.4939|0.4939|
|SingleUnsigned|mse|5|2.29757e-07|[-1.35016e-07,7.18628e-07]|0.2887|1.0000|
|SingleUnsigned|energy_score|5|-0.000258257|[-0.000420166,-0.000136165]|0.0022|0.0066|
|NoStateSupervision|mse|5|2.20721e-07|[-2.13726e-07,8.66723e-07]|0.4305|1.0000|
|NoStateSupervision|energy_score|5|-0.000193296|[-0.000284053,-0.000105579]|0.0002|0.0008|

## State metrics and provenance

state_metrics.csv reports future-lead concept MAE, matched concept persistence MAE, graph IoU and graph edge MAE. concept_coordinates.csv retains every coordinate/lead. Anchored h=0 is excluded. Graph support uses strict |A_ij|>0.05 off the diagonal; an empty union has IoU0. These fidelity measures do not identify a causal market graph.

All input completion files, loss archives, state tables and available configuration/audit records are SHA256-hashed in summary.json. With --exact-es, full prediction archives are also hashed and the exact finite-scenario ES is recomputed in float64. Without that flag, stored ES retains its original numerical precision; point losses are always recomputed in float64.
