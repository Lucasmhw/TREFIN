# Fresh official-upstream supplementary experiments

These are new task-trained experiments, distinct from the authentic historical separate-host main-table runs. No historical numerical result is used in fitting, selecting or calibrating these experiments.

## Sources

- TEDM: https://gitlab.com/dlr-dw/tedm, revision `1d65888cc1ead587f738a0081a150d3c15d97b17`.
- Timer-XL: https://github.com/thuml/OpenLTM, revision `0b3005099d380ecc00d512f72553c7fe8fccca99`.

Only required source files are included, unchanged from upstream, together with their licenses. The manifest records every source hash. All adaptations live in `fresh_official_baselines.py`.

## Data and splits

The compact NPZ stores raw one-period excess log returns, existing forecast-origin indices, exact archived targets and exact train/validation/test membership for fold 1. It has 2,135 training origins, 498 validation origins and 500 test origins, 32 assets and five forecast leads. Reconstruction from the archived raw panel and factor/risk-free file was checked against every stored target with exact float32 equality. Raw file hashes match the historical model archive. Both baselines receive 126 observed return rows; no future measurements enter their input.

Normalization uses each asset's mean and population SD on unique raw training rows through the last admissible training label. The initialized first-row return is excluded from normalization fitting. The factor file provides the risk-free subtraction required by the task; FF5 predictor values, concepts, price/volume channels and graphs are not baseline inputs.

## Models and adaptations

**Timer-XL:** unchanged official `Model`, 128-dimensional embeddings, three transformer layers, four heads, feed-forward width 512, dropout 0.1 and original instance normalization. Six input tokens of 21 rows cover all 126 observations. The configurable output token has length five; its final token forecasts the five targets. Training minimizes last-token MSE. This is task training from random initialization, not pretraining or fine-tuning. Joint scenarios add centered training residual paths to point forecasts; this explicit adapter is not described as an original probabilistic Timer-XL head.

**TEDM:** unchanged official UNet, TEDM noise schedule, preconditioner and one-step stochastic sampler. The UNet uses 112 temporal positions, 32 base channels, two residual blocks per resolution, channel multipliers [1,2,3,4,5], attention at [112,56,28,14], alias-free resampling and rotary attention. A 13-row conditioning offset plus one padding row gives 126 total observed rows at forecast time. Five native rollout steps are performed, and the final five coordinates are evaluated. Training uses the official weighted denoising loss on 126-row training windows ending at each admissible five-step training target. EMA uses the official `ema-pytorch` package with decay 0.995 and update interval 10. Scenarios are native stochastic 32-asset outputs; their Monte Carlo mean supplies the point forecast.

The TEDM generic upstream template supplies the declared noise parameters; window lengths and attention resolutions are adapted to this task. The source uses a track-major `repeat` followed by a final reshape that does not preserve track/origin axes. The wrapper implements the same sample equations, clamping and rolling updates but unpacks `[S*B,C,L]` as `[S,B,L,C]`. Tests verify independent track/origin sentinels, exact equivalence to upstream deterministic rollout and exact checkpoint reload.

## Selection and budget

For each model and each seed [1,7,19,37,73], two candidates use learning rates 0.001 and 0.0003 with identical initialization and shuffled-example seed. Batch size is 64. The cap is 100 complete data epochs, with patience 15 validation checks. Actual optimizer steps and wall time are recorded separately. This distinction matters because TEDM's upstream variable `max_epochs` counts optimizer steps rather than complete data passes.

Timer-XL uses Adam and the upstream cosine scheduling family. TEDM uses Adam with betas (0.9,0.96) and the unchanged native plateau scheduler (100-update warmup/hold, factor 0.9, patience 15, relative threshold 0.01). Both use gradient clipping at 1.0. Checkpoint and candidate selection minimize validation energy score, using 64 fixed scenarios on all validation origins after every epoch. Test outcomes do not influence training, selection or calibration.

After selection, a fresh instance reloads the checkpoint and must reproduce its validation ES. The same scalar mean-calibration family as TREFIN, rho in [0.05,1], is fitted by validation ES while scenario innovations remain fixed. The test uses 256 scenarios. Before/after MSE, RMSE, MAE, ES, msIC, msIR, 90% coverage and interval width are recorded. Separate per-origin losses enable paired temporal tests. Saved means and native scenarios are not inferred from target metrics.

## Execution

Use the existing PyTorch/CUDA environment; add only the small dependencies in `requirements-supplement.txt`.

```bash
python -m pip install -r requirements-supplement.txt
python fresh_official_baselines.py --family timerxl --repo /path/f-concord --bundle . --out /path/smoke_timerxl --device cuda --smoke
python fresh_official_baselines.py --family tedm --repo /path/f-concord --bundle . --out /path/smoke_tedm --device cuda --smoke
python fresh_official_baselines.py --family timerxl --repo /path/f-concord --bundle . --out /path/fresh_timerxl --device cuda
python fresh_official_baselines.py --family tedm --repo /path/f-concord --bundle . --out /path/fresh_tedm --device cuda
```

Use one model family per process because the upstream repositories both use generic Python package names. Model training should run on the authorized server. CPU smoke tests use only two training examples and three scenarios, and are not evidence of forecasting performance.

Every seed produces `result.json`, calibration metadata, candidate histories, best/last checkpoints, validation and test forecasts, and a small `paired_losses.npz`. `summary.json` is written only when all requested seeds finish. A completed candidate/seed is reused on restart; an interrupted candidate is retrained deterministically. Protocol changes require a new output directory.
