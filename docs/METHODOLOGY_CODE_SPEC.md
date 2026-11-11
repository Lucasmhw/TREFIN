# F-CONCORD code-normative methodology specification

## Normative-source policy

This repository is the normative source for the paper Methodology section. The Chinese
manual remains the design rationale and theorem reference, but a claim is allowed only when
the versioned code, resolved configuration and saved audit artifacts implement it. The
priority order is:

1. executed source code plus `config_resolved.yaml`;
2. run-local manifests, arrays, logs and audit reports;
3. `docs/manual_contract.yaml` at the snapshot saved inside the run;
4. the earlier PDF manual as design background.

No paper equation may describe a different target, split, normalization population, graph,
decoder or test protocol from the executed code. If the design changes, this file and the
machine-readable contract must change before new headline experiments are launched.

## Frozen current model contract

- **Supervised target:** for forecast origin `t`, the main label is the joint path of
  per-asset excess log returns at leads `h=1,...,H`. Concept and graph state paths at
  `h=0,...,H` are auxiliary supervised labels. Inference receives only the measured
  `(C_t, A_t, s_t)` at the forecast origin.
- **Concept order:** momentum, log upside semivariance, log downside semivariance, jump
  share and log Amihud illiquidity at scales `[5,21,63,126]`, followed by one order-flow
  coordinate and the rolling factor betas. The order never changes inside an experiment.
- **Order flow:** daily data without signed trades use the explicitly weak signed-dollar-
  volume return proxy. BTCF uses the observed interval taker-buy imbalance. These two
  measurement operators share a semantic slot but are never presented as the same raw
  statistic.
- **Factor residual graph:** weighted rolling ridge betas and residuals are re-estimated at
  each origin. Graphical-Lasso regularization is selected from training data only using
  next-step Gaussian likelihood plus edge instability. Smoothed signed residual partial
  correlations form the graph.
- **Market state:** the three immutable coordinates are market log realized variance, mean
  absolute residual partial correlation, and weighted market log illiquidity.
- **Fold statistics:** median/MAD fallbacks, coordinate clipping bounds and the graph norm
  bound are fitted only on current states at purged training forecast origins. Overlapping
  future state labels are excluded from scaler fitting and are clipped to the resulting
  compact state sets.
- **Strict bottleneck:** raw prices, volumes and factor arrays do not enter the neural
  dynamics or decoder. The model sees only named concept, graph and market states.
- **Dynamics:** direction-valued concepts use the signed Laplacian; magnitude-valued
  concepts use the unsigned Laplacian. Concept and graph updates are computed in parallel
  from the old state and projected into the train-defined compact sets. The bounded
  nonlinear residual is a same-capacity SiLU MLP, not a KAN.
- **Distribution:** every lead returns one joint multi-asset distribution with an SPD graph
  precision, unit-variance multivariate Student-t base shock, asymmetric positive/negative
  scales, and the exact analytic mean correction.
- **Predictable-signal gate:** the calibrated configuration writes every scenario as
  `Y^(s) = mu_struct + epsilon^(s)` and replaces it after checkpoint selection by
  `Y_tilde^(s) = rho * mu_struct + epsilon^(s)`, with `rho` constrained to `[0.05,1]`.
  The scalar is fitted on validation scenarios only by bounded minimization of Energy Score
  and is frozen in both `config_resolved.yaml` and `best.pt` before a fresh-model reload.
  The same shift is applied to every scenario, so pairwise scenario distances, dependence,
  tail innovations and the Energy Score dispersion term are exactly invariant. The remaining
  validation objective is a sum of Euclidean norms of affine scalar functions and is convex;
  the stored optimization is therefore a global one-dimensional optimum. Strict positivity
  preserves every nonzero forecast sign and positive-scale msIC interpretation.
- **Optimization:** the exact finite-sample all-pairs energy score is combined with
  normalized concept-path, graph-path, monotonicity and local small-gain penalties.
- **Selection and final test:** every epoch logs training and validation metrics only,
  including MSE, RMSE and MAE on decimal excess-log-return labels. The
  best checkpoint is chosen solely by validation energy score. When signal calibration is
  enabled, `rho` is then fitted using that checkpoint's validation scenarios without test
  access. The calibrated configuration and checkpoint are loaded into a fresh model, checked
  for prediction equality, and only then evaluated once on the final test split.
- **Percentage-error policy:** MAPE is excluded because realized returns may be zero or
  arbitrarily close to zero. The scale-aware supplementary statistic is out-of-sample
  `R2 = 1 - MSE_model/MSE_ZeroReturn`, deterministically derived from frozen test predictions.
- **Stability claim:** algebraic invariants, the global structural gain bound and sampled
  JVP/VJP Jacobian diagnostics are reported separately. A sampled local radius below one
  never substitutes for a failed global theorem premise.

## Audited empirical adaptations

### FinMultiTime US

The public numeric archive does not provide historical index membership, delisting returns
or first-release factor vintages. The implemented task is therefore a fixed **training-era
liquidity-selected US equity cohort**, not a survivorship-free historical S&P 500 study.
This limitation is stored in the source manifest and must appear in the paper.

### FinTSBridge BTCF

The repository downloads the exact 43,014-row `BTCF.csv` released in the official
FinTSBridge code artifact. FinTSBridge stores
`Z_t = log(P_t/P_0)+100`; F-CONCORD constructs `exp(Z_t-Z_0)`, so its log-return label is
exactly `Z_t-Z_{t-1}`. Log base volume and taker-buy volume are inverted with `expm1`.
Because the original absolute price normalization is unavailable, dollar volume is known
only up to an instrument-specific positive constant. Train-only coordinate standardization
removes that fixed scale, but the limitation is still declared in the manifest. The causal
common-BTC factor is the contemporaneous equal-weight spot/futures log return; it separates
shared BTC price discovery from residual basis dependence and uses no future label.

The official benchmark has 73 non-unit hourly gaps. The current code follows its observed
sequence ordering, so a lead denotes the next observed row rather than guaranteed wall-clock
hours across a gap. This is an intentional, recorded protocol choice.

The executable config adopts the official raw-row split sizes 30,109/4,303/8,602. Their
timestamp boundaries are stored explicitly rather than recomputed from floating-point
fractions. Since each F-CONCORD origin has a five-row future path label, an origin is retained
only when its complete label lies before the next official split start. Consequently the
effective origin counts are smaller at each left/right edge, while the official row ownership
is unchanged and no target crosses train/validation or validation/test.

For msIC/msIR, the paper text says “rank correlation” but the released `f_tools.py` computes
population Pearson correlation along the forecast horizon. The repository follows the
released code so published comparisons are commensurable, reports constant-path coverage,
and uses zero correlation for a path whose denominator is numerically zero.

## Computational adaptation

Causal state preparation is deterministic and independent of the neural random seed. Real
data runs cache it under a SHA-256 key containing the panel/factor hashes, complete data
configuration and hashes of `concepts.py`, `graph.py` and `dataset.py`. Each rolling fold has
its own key, while its five neural seeds reuse the same prepared states. Every run copies the
cached arrays into its own audit directory, so reuse changes runtime only, not statistical
semantics or artifact independence.

For daily panels the residual-dependence graph is treated as a slower structural state and
is refitted every five observed trading rows; the latest fitted graph is carried forward on
intervening rows. Factor beta, asset concepts, raw correlation and market-return quantities
are still recomputed at every daily origin. Carry-forward is strictly causal and avoids both
future interpolation and repeated high-variance Graphical-Lasso fits on nearly identical
126-day windows. Intraday BTCF uses `graph_refit_stride=1`. The cadence is part of the
resolved configuration, prepared-state cache key and graph audit (`refit_count`, refit rate,
fit-only convergence/fallback statistics); changing it defines a different experiment.

Graphical Lasso uses coordinate descent with at most 500 outer iterations and a dual-gap
tolerance of `1e-4`. These values are explicit configuration, not test-tuned parameters. A
non-converged or non-SPD fit is retried at stronger regularization and finally replaced by a
declared ridge precision estimate; every retry and fallback remains in the run audit.

## Completed fixed-cohort rolling protocol

The registered FinMultiTime US-32 `H=5` protocol completed three ordered, non-overlapping
test folds with seeds `[1, 7, 19, 37, 73]`. Split membership requires every return/state
label through lead `H` to remain inside its own registered end date. Each epoch evaluates
train and validation only; the test set is called exactly once after validation selection
and a fresh checkpoint reload. `scripts/audit_protocol.py` independently reopens every
checkpoint and evidence file. This closes the execution requirement E04, but it does not
repair the cohort's missing point-in-time membership/delisting data or supply the unfinished
datasets and equal-budget baselines listed below.

The earlier completed 15-run protocol used `rho=1` and remains an immutable reference
artifact. The separate calibrated protocol uses
`configs/finmultitime_us_calibrated_protocol.yaml`; it is a new experiment and must not be
presented as completed until its own 15-run audit passes.

## Claims still prohibited

Until the corresponding saved protocol is complete, the repository must not claim:

- a completed three-layer ICLR benchmark;
- point-in-time, survivorship-free FinMultiTime equity evidence;
- a global contraction theorem when the structural radius is at least one;
- limit-order-book validity from daily or hourly aggregate order-flow proxies;
- state-of-the-art performance without the full equal-budget baseline and multi-seed matrix.
