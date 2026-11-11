# Matched structural retraining summary

Status: **complete**. Completed 25/25 requested runs.

All five variants are retained. Means and sample SD describe training-seed variation. Paired inference averages matched seed losses within each chronological origin; all assets and five leads remain grouped. Positive candidate-minus-Full loss differences favor Full. Primary circular block length is20;5 and10 are sensitivity checks. MSE and ES p values are Holm-adjusted over the four variants separately within each fold/block. MAE p values are unadjusted.

NoGraph removes the A input/rollout/decoder channel and graph-target loss. It retains the market state, including its residual-dependence summary; it does not remove every graph-derived quantity.

## Fold1 calibrated results

MSE is multiplied by10^4; all other displayed metrics are multiplied by10^2. RMSE is computed for each run before seed aggregation.

| Variant | Seeds | MSE | RMSE | MAE | ES | msIC | msIR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 5 | 6.8174 ± 0.0065 | 2.6110 ± 0.0012 | 1.6703 ± 0.0003 | 8.9752 ± 0.0180 | 0.5719 ± 0.2736 | 1.1623 ± 0.5600 |
| NoGraph | 5 | 6.8268 ± 0.0092 | 2.6128 ± 0.0018 | 1.6717 ± 0.0011 | 9.0569 ± 0.0297 | 0.7127 ± 0.4141 | 1.4379 ± 0.8284 |
| SingleSigned | 5 | 6.8176 ± 0.0064 | 2.6110 ± 0.0012 | 1.6703 ± 0.0003 | 8.9746 ± 0.0183 | 0.5883 ± 0.2876 | 1.1933 ± 0.5870 |
| SingleUnsigned | 5 | 6.8185 ± 0.0068 | 2.6112 ± 0.0013 | 1.6704 ± 0.0003 | 8.9695 ± 0.0315 | 0.6537 ± 0.3145 | 1.3217 ± 0.6373 |
| NoStateSupervision | 5 | 6.8211 ± 0.0023 | 2.6117 ± 0.0004 | 1.6705 ± 0.0003 | 8.9714 ± 0.0321 | 0.2665 ± 0.3826 | 0.5441 ± 0.7676 |

Primary paired comparisons (raw loss units; no benefit claim is inferred from ranking alone):

| Candidate − Full | Metric | Matched seeds | Mean difference |95% CI|Raw p|Holm p|
|---|---|---:|---:|---|---:|---:|
|NoGraph|mse|5|9.37564e-07|[-7.37816e-07,2.85136e-06]|0.3035|0.9106|
|NoGraph|energy_score|5|0.000816254|[0.000407014,0.00138892]|0.0054|0.0162|
|SingleSigned|mse|5|1.42947e-08|[-8.53673e-09,5.21057e-08]|0.3733|0.9106|
|SingleSigned|energy_score|5|-6.02603e-06|[-1.09269e-05,-1.41679e-06]|0.0174|0.0348|
|SingleUnsigned|mse|5|1.08735e-07|[-2.96995e-08,2.9516e-07]|0.1796|0.7183|
|SingleUnsigned|energy_score|5|-5.76349e-05|[-8.75432e-05,-3.46782e-05]|0.0004|0.0016|
|NoStateSupervision|mse|5|3.6422e-07|[-5.0364e-07,1.30015e-06]|0.3721|0.9106|
|NoStateSupervision|energy_score|5|-3.87704e-05|[-0.000117843,4.29719e-05]|0.3321|0.3321|

## State metrics and provenance

state_metrics.csv reports future-lead concept MAE, matched concept persistence MAE, graph IoU and graph edge MAE. concept_coordinates.csv retains every coordinate/lead. Anchored h=0 is excluded. Graph support uses strict |A_ij|>0.05 off the diagonal; an empty union has IoU0. These fidelity measures do not identify a causal market graph.

All input completion files, loss archives, state tables and available configuration/audit records are SHA256-hashed in summary.json. With --exact-es, full prediction archives are also hashed and the exact finite-scenario ES is recomputed in float64. Without that flag, stored ES retains its original numerical precision; point losses are always recomputed in float64.
