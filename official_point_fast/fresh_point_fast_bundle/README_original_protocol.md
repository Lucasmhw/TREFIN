# Fresh TimeXer and PyPOTS TimeMixer++ supplemental experiments

These are new task-trained experiments on the existing exact fold-1 targets, not reconstructions of the historical separate-host results. No historical result is used for fitting or selection. Use one family per process.

## Pinned sources and licenses

- **TimeXer (official THUML/TSLib implementation)**: `https://github.com/thuml/Time-Series-Library`, revision `4e938a1767106324dd753b2a44832bf870a0252e`, MIT. Its `models/TimeXer.py` content equals the standalone author's `thuml/TimeXer` revision `76011909357972bd55a27adba2e1be994d81b327`. TSLib supplies an explicit MIT license. Only the necessary source modules are included, unchanged.
- **TimeMixer++ (PyPOTS implementation)**: `https://github.com/WenjieDu/PyPOTS`, revision `2271533587bab250163437ed39cfe13cefd8b80a`, BSD-3-Clause. This is explicitly the maintained PyPOTS implementation, not an author-source reproduction. Its source cites `https://anonymous.4open.science/r/TimeMixerPP`; that author's review repository returned `repository_expired` on 8 November 2026. The author-linked `kwuking/TimeMixer` repository contains original TimeMixer and was not substituted. The PyPOTS implementation and native-head behavior are documented by its maintainers at `https://github.com/WenjieDu/PyPOTS/issues/789`.

The manifest pins and hashes every bundled upstream file, including both licenses. The runner checks source, adapter and helper hashes before training. PyPOTS imports are restricted to the unchanged mathematical modules; namespace packages avoid importing its unrelated dataset/training stacks. No math module is patched. The frozen common helper file is an exact copy of the already audited fresh TEDM/Timer runner and is used for data loading, point inference, centered residual scenarios and ES only; its `main()` is never invoked.

## Task and architecture configuration

Both methods observe all 126 past one-period excess log returns for all 32 assets, and forecast the next five return vectors. They use the existing `official_supplement_bundle/fold1_raw_returns.npz` by path, with exactly matching data hash, 2,135/498/500 train/validation/test origins, and the existing purged splits. Per-asset mean and population SD are fitted only on the archived admissible training rows. No FF5 predictor, volume, concept, graph or future calendar input is supplied. TREFIN's richer concept/factor/volume predictor representation must therefore be disclosed when comparing results; this is not an equal-information architecture ablation.

**TimeXer:** native multivariate `features=M` path, 21-day patches covering all 126 rows exactly, 128-dimensional model, 512-dimensional feed-forward width, three encoder layers, four heads, dropout 0.1, GELU, native input normalization and five-step prediction head. Asset return histories supply both endogenous patch/global tokens and the native variate embeddings. No external covariate is fabricated. Registered parameters: 821,509.

**PyPOTS TimeMixer++:** one fixed architecture: `d_model=32`, `d_ffn=64`, two mixer blocks, three dominant periods, three inception kernels, channel independence and channel mixing enabled, two average-downsampling layers with factor three. This gives exact scales 126/42/14 with no padding or truncated observations. It retains frequency-based time imaging, dual-axis attention, multiscale trend/season mixing, multiresolution mixing and native RevIN. Its forecast returns the final/coarsest prediction head, exactly as the pinned upstream source; the adapter does not silently replace this with head averaging. Registered parameters: 600,306.

Native PyPOTS period selection averages over the batch. **At inference the wrapper calls native forecast separately for each origin**, so a later forecast origin's observed history cannot influence an earlier origin. Training retains ordinary shuffled minibatches of 64. Native modules, their equations and learned parameters are unchanged. This per-origin evaluation also applies when constructing the training residual pool. CPU tests verify identical predictions with outer inference batch sizes one and two. The extra Python loop can increase inference time and is recorded in the protocol.

## Training, selection, scenarios and calibration

For each family, five seeds `[1,7,19,37,73]` and two learning-rate candidates `[0.001,0.0003]` are fixed before test evaluation. There is **one architecture per family**: 32/64 denotes TimeMixer++ model/feed-forward widths, not two candidates. Training uses MSE, Adam betas `(0.9,0.999)`, no weight decay, gradient clipping at 1, batch size 64, epoch cosine LR down to `1e-8`, cap 100 full data epochs and patience 15 validation checks. Candidate initialization and shuffle seed match within each seed. This is the same candidate/epoch/calibration budget as fresh Timer-XL; it does not claim equal architecture sizes or training budgets with every historical method.

Checkpoint and candidate selection minimize **uncalibrated validation ES**, with 64 fixed residual scenarios on all validation origins. Residual scenarios sample complete `[5,32]` training-residual paths after centering each lead/asset residual mean. This preserves empirical path dependence and makes the backbone output the empirical adapter's predictive mean. It is an explicit probabilistic adapter, not an original native probabilistic head of either architecture.

After selection, a newly instantiated model reloads the checkpoint and must reproduce the selected validation ES. A single positive `rho` in `[0.05,1]` is fitted by validation ES with fixed residual innovations. The selected candidate and calibration JSONs are persisted before test inference. Test uses 256 scenarios. Point metrics, exact off-diagonal cross-sectional ES, 90% coverage and width are saved before and after calibration, together with per-origin losses and target hashes. Test outcomes do not affect model fitting, selection or calibration. The data loader contains all split arrays; the claim is absence of test-outcome use, not physical absence of test data from memory.

## Installation and execution

Use the existing PyTorch/CUDA environment. The small included universal wheels are import dependencies of an unused Reformer class imported by the unchanged TSLib attention module; TimeXer itself uses full attention. PyPOTS mathematical modules require only existing PyTorch/NumPy.

```bash
python -m pip install --no-index --no-deps --find-links wheels -r requirements-point.txt

python fresh_point_baselines.py --family timexer --repo /path/f-concord --bundle . --data-bundle /path/official_supplement_bundle --out /path/smoke_timexer --device cuda --smoke
python fresh_point_baselines.py --family timemixerpp --repo /path/f-concord --bundle . --data-bundle /path/official_supplement_bundle --out /path/smoke_timemixerpp --device cuda --smoke

python fresh_point_baselines.py --family timexer --repo /path/f-concord --bundle . --data-bundle /path/official_supplement_bundle --out /path/fresh_timexer --device cuda
python fresh_point_baselines.py --family timemixerpp --repo /path/f-concord --bundle . --data-bundle /path/official_supplement_bundle --out /path/fresh_timemixerpp --device cuda
```

The pipeline layout matches the existing fresh-results summarizer: root `protocol.json`, then `seed_*/result.json`, `paired_losses.npz`, selected candidate, calibration, training histories, checkpoints, validation/test predictions and a final `summary.json`. Protocol changes require a new output directory. Completed candidates/seeds are reused; an interrupted candidate restarts from initialization. Every reported result is computed from real forecasts.

CPU smoke reports under `tests/` confirm finite real-data training loss, nonzero gradients, exact reload, independent-origin evaluation, expected `[B,5,32]` point/`[S,B,5,32]` scenario shapes and zero ES discrepancy. They use two training examples and three scenarios and are **not forecasting-performance evidence**. Root should run the CUDA smoke tests before launching full experiments.
