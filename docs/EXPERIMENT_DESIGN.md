# Complete experimental design

## 1. Falsifiable research hypotheses

- **H1 - Predictable-characteristic sufficiency.** Conditional on current concepts, residual graph and market state, a raw-history probe should provide little strictly out-of-sample distribution-score improvement.
- **H2 - Common-factor/residual separation.** A graph estimated after factor residualization should improve graph stability and test distribution scores relative to a raw-return graph.
- **H3 - Typed propagation.** Signed propagation for directional concepts and unsigned propagation for magnitude concepts should outperform all-signed and all-unsigned alternatives.
- **H4 - Closed-loop value.** Joint concept-graph evolution should outperform fixed-graph and graph-free variants.
- **H5 - Distributional value.** The asymmetric heavy-tail joint decoder should improve proper scores, tail calibration and cost-adjusted decisions over point or symmetric Gaussian decoders.
- **H6 - Interpretability/stability compatibility.** Named interventions should be directionally stable while the complete two-block dynamics satisfy or approach the preregistered small-gain condition.
- **H7 - Weak-signal calibration.** In low signal-to-noise return regimes, a strictly positive validation-calibrated gate on the structural conditional mean should improve proper distribution scores and point errors without changing scenario innovations, dependence, forecast signs or positive-scale information coefficients.

Failure of H1, H2 or H3 is evidence against the central representation, not merely a tuning failure.

## 2. Dataset tiers

### Tier A: daily cross-market main study

Use the numerical core of **FinMultiTime** (2025), covering U.S. and Chinese equities. The
paper reports 5,105 stocks, 2009-2025, multiple frequencies and four modalities. The first
F-CONCORD paper should use only auditable point-in-time numerical inputs; text/table/chart
modalities are a predeclared extension.

- Paper: https://arxiv.org/abs/2506.05019
- Freeze the exact data snapshot and official processing commit in the run manifest.

### Tier B: multi-market/multi-horizon external study

Use **FinTSBridge** (2025) GSMI, OPTION and BTC tasks to test global indices, derivatives,
crypto spot/futures and multiple horizons.

- Paper: https://arxiv.org/abs/2503.06928
- Treat it as an evaluation suite; record the exact code/data snapshot.

### Tier C: microstructure external study

Use the public portions of the **TLOB** suite, prioritizing newer Bitcoin LOB data and using
FI-2010 as a compatibility benchmark. Restricted LOBSTER assets may only be external checks.

- Official code: https://github.com/LeonardoBerti00/TLOB
- Paper: https://arxiv.org/abs/2502.15757

### Tier D: practical financial benchmark alignment

Use **FinTSB** for ranking, portfolio and prediction-error protocols with fees and regulatory
constraints.

- Official code: https://github.com/TongjiFinLab/FinTSBenchmark
- Paper: https://arxiv.org/abs/2502.18834

No benchmark is considered reproducible until raw licenses, membership history, delistings,
corporate-action adjustments, time zones and publication timestamps pass the audit.

## 3. Forecast tasks

1. Joint next-period return distribution for all assets.
2. Multi-horizon path forecasting with step returns, not cumulative labels.
3. Cross-sectional ranking and long-short portfolio construction.
4. VaR and Expected Shortfall calibration for equal-weight and factor-neutral portfolios.
5. LOB direction classification as a separate decoder in the microstructure study; do not mix its labels with daily return-distribution results.

Primary daily horizons: `H in {1,5,21,63}`. The debug configuration uses `H=3` only to test code.

## 4. Chronological protocol

1. Use expanding or rolling forecast origins.
2. Fit all normalizers, factor hyperparameters, graph regularization and model hyperparameters only inside the training period.
3. Purge overlapping labels and apply an embargo of at least the maximum horizon between training and validation/test.
4. Use point-in-time constituents and include delisted returns.
5. Align financial/fundamental data by first-release timestamp; never use revised values retroactively.
6. Freeze at least three continuous test blocks: calm, high-volatility and crisis/stress.
7. Use at least five random seeds for trainable models and report mean, standard deviation and per-seed artifacts.
8. When predictable-signal calibration is enabled, select the neural checkpoint first, fit the positive scalar gate on validation scenarios only, freeze it in the resolved config/checkpoint, reload a fresh model, and access test only after those steps.

## 5. Baseline matrix

### Statistical and financial controls

- zero-return/random-walk forecast;
- historical mean and volatility;
- AR/VAR with shrinkage;
- GARCH/DCC-GARCH for conditional volatility/dependence;
- Elastic Net and gradient-boosted trees;
- factor-only and factor-plus-residual-linear models.

### General neural forecasting

- DLinear/NLinear;
- PatchTST official implementation: https://github.com/yuqinie98/PatchTST
- iTransformer official implementation: https://github.com/thuml/iTransformer
- TimeMixer/TimeMixer++ official implementation: https://github.com/kwuking/TimeMixer
- a parameter-matched MLP/GRU and Transformer.

### Financial specialized

- MASTER for market-guided stock representation;
- FinTSB-provided backbones under its standardized protocol;
- DeepLOB, BinBTabL and TLOB for order-book experiments.

### Foundation models

- TimesFM zero-shot and permitted fine-tuning: https://github.com/google-research/timesfm
- Chronos zero-shot and permitted fine-tuning: https://github.com/amazon-science/chronos-forecasting
- for 2026 runs, also preregister TimesFM 2.x/2.5 and Chronos-2 if public weights and compatible licenses remain available.

Every baseline receives the same forecast origins, inputs available at that origin, tuning
budget, transaction costs and evaluation code. Foundation-model results distinguish zero-shot
from fine-tuned runs.

## 6. Metrics

### Point and rank

- MSE, MAE, RMSE, directional accuracy;
- out-of-sample R2 against the ZeroReturn forecast, defined as
  `1 - MSE_model / MSE_ZeroReturn`;
- no MAPE: daily returns can be zero or arbitrarily close to zero, so percentage errors have
  an unstable denominator and a misleading scale;
- cross-sectional Pearson IC and Spearman rank IC;
- top-minus-bottom return spread.

### Distribution and tails

- multivariate energy score (primary);
- CRPS for marginals;
- pinball loss at 1%, 5%, 50%, 95%, 99%;
- empirical coverage and interval width;
- PIT and rank histograms;
- VaR unconditional coverage, independence and conditional-coverage tests;
- Expected Shortfall backtests.

### Decision value

- annualized return, volatility, Sharpe, Sortino, maximum drawdown;
- turnover, capacity proxies and net performance at several fee/slippage levels;
- mean-variance and CVaR portfolio regret relative to an oracle using the realized path.

### Interpretability and structure

- concept MAE/R2 by coordinate, horizon, asset and regime;
- seed-to-seed concept fingerprint correlation;
- edge sign accuracy, Jaccard, density, degree and spectral stability;
- residual-probe improvement;
- intervention direction, magnitude and bound tightness;
- empirical block-Jacobian norms and small-gain certificate status.

## 7. Ablations

Required ablations are: no concepts/raw skip, no graph, raw-correlation graph, fixed graph,
unsigned-only graph, signed-only graph, no market state, latent market state, no forward
concept supervision, Gaussian symmetric decoder, symmetric Student-t decoder, point decoder,
no mean correction, no monotonicity, no gain penalty, MLP in place of KAN-like additive
response, and factor set sensitivity.

The repository now implements true retraining protocols for `NoGraph`, `NoMarketState`,
`NoRiskConcepts`, `NoForwardConceptSupervision` and `GaussianSymmetricDecoder`. These are
structural training-time removals, not the older post-hoc input interventions. The remaining
extended ablations above are still required for a fully saturated paper appendix.

## 8. Statistical tests

- Diebold-Mariano with HAC correction for overlapping horizons;
- Model Confidence Set or SPA/Reality Check for many-model comparisons;
- stationary/block bootstrap for Sharpe, drawdown and portfolio differences;
- paired seed/bootstrap intervals for concept and graph fidelity;
- Holm or Benjamini-Hochberg correction for families of secondary hypotheses.

Report effect sizes and confidence intervals, not only p-values.

The executable statistical layer includes overlap-aware DM--HAC, paired circular-block
confidence intervals, Holm family-wise correction, a studentized block-bootstrap SPA test
and sequential range-statistic MCS. All operate on loss at the forecast-origin level.

## 9. Hyperparameter protocol

Use a fixed search budget per trainable baseline. Optimize validation energy score first,
then break ties with concept fidelity and parameter count. Never tune on test Sharpe. Record
all candidate configurations, failed runs and wall-clock/GPU usage. Important F-CONCORD axes:

- concept scales and graph window;
- ridge and Graphical-Lasso strength;
- concept/graph auxiliary weights;
- graph refresh and diffusion bounds;
- hidden width and lead embedding;
- Student-t sample count;
- positive predictable-signal calibration bounds (fixed before test; the scalar itself is fit on validation only);
- monotonic and gain regularization.

## 10. Figure plan

`visualization.py` generates 50 actual-data figures grouped into optimization, forecasting,
calibration, interpretability, graph structure, theory/stability, distribution, robustness,
decision value and systems performance. `figures/manifest.json` is the machine-readable
index. No figure may be manually edited without retaining its generating script and source
run identifier.
