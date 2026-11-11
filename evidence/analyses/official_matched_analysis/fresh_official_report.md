# Matched TEDM and Timer-XL comparisons

The comparison uses the pinned official backbones and the task adaptations documented in the accompanying protocol.

The five fixed training seeds are 1, 7, 19, 37 and 73. Both methods use the exact fold-1 origins and targets; checkpoint and learning-rate candidate selection use validation ES only. The same positive mean-calibration family is then fitted on validation scenarios. Results below are completed runs, with mean ± sample SD over seeds.

| Method | Stage | MSE x10^4 | RMSE x10^2 | MAE x10^2 | ES x10^2 |
|---|---|---:|---:|---:|---:|
| TEDM | before mean calibration | 11.5255 ± 0.0142 | 3.3949 ± 0.0021 | 2.2967 ± 0.0006 | 14.8430 ± 0.0031 |
| TEDM | after mean calibration | 6.8296 ± 0.0003 | 2.6133 ± 0.0001 | 1.6743 ± 0.0000 | 10.5919 ± 0.0005 |
| Timer-XL | before mean calibration | 6.9732 ± 0.0247 | 2.6407 ± 0.0047 | 1.6973 ± 0.0038 | 9.3447 ± 0.0184 |
| Timer-XL | after mean calibration | 6.8352 ± 0.0008 | 2.6144 ± 0.0002 | 1.6718 ± 0.0001 | 9.2117 ± 0.0026 |
| TREFIN | before mean calibration | 6.8542 ± 0.0414 | 2.6180 ± 0.0079 | 1.6764 ± 0.0066 | 9.0060 ± 0.0359 |
| TREFIN | after mean calibration | 6.8171 ± 0.0066 | 2.6110 ± 0.0013 | 1.6704 ± 0.0003 | 8.9744 ± 0.0160 |

## Paired temporal tests

Losses are averaged over the five seeds at each of the 500 identical test origins before temporal inference. Positive differences favor TREFIN. Primary circular-block bootstrap intervals use blocks of 20 origins and 10,000 resamples; two-sided p values use null-centered bootstrap means with a +1 correction. Blocks of five and ten origins and DM with HAC lag 20 provide sensitivity checks. Holm adjustment is across the two comparisons within each loss and block choice. TREFIN energy scores are recomputed from its original scenarios using float64 exact pairwise distances.

| Comparator | Loss | Difference | 95% block-20 interval | Holm-adjusted p |
|---|---|---:|---:|---:|
| TEDM | MSE | 1.2473355e-06 | [-4.8265362e-06, 6.326853e-06] | 0.6541 |
| TEDM | MAE | 3.9307514e-05 | [5.4184182e-06, 7.6055272e-05] | 0.05979 |
| TEDM | ES | 0.016175314 | [0.013307035, 0.019914255] | 0.0002 |
| Timer-XL | MSE | 1.8121147e-06 | [-1.3429632e-06, 5.4387583e-06] | 0.5575 |
| Timer-XL | MAE | 1.4408609e-05 | [-1.2578119e-05, 4.4950593e-05] | 0.3192 |
| Timer-XL | ES | 0.0023737209 | [0.00041320236, 0.0052344103] | 0.047 |

## Runtime and calibration

| Method | Mean selected rho | Total optimizer updates (both candidates, all seeds) | Total candidate training+validation seconds |
|---|---:|---:|---:|
| TEDM | 0.05000 | 19720 | 22837.6 |
| Timer-XL | 0.05000 | 6120 | 379.1 |

All target hashes, test indices, saved per-origin loss means, checkpoint reload comparisons and calibration artifacts were checked. Timer-XL uses the specified training-residual scenario adapter; TEDM uses native stochastic samples. The inputs, normalization, heads, objectives, selection budgets and calibration settings are recorded in the accompanying protocol.
