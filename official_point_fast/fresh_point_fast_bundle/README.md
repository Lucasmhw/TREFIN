# Equivalent vectorized independent-origin TimeMixer++ evaluation

This separate candidate bundle preserves the original model, weights, loss, training execution, optimizer, search budget, splits, residual scenario adapter and calibration. It changes evaluation scheduling only. The original protocol and source attribution are in `README_original_protocol.md`; no running original source is overwritten. The original runner remains bundled as the numerical reference. Start an optimized experiment in a new output directory after the CUDA checks pass.

## Why the grouping is equivalent

PyPOTS first expands its channel-independent rows in origin-major order: `[B,32,T,D]` becomes `[B*32,T,D]`. Native FFT frequency selection averages this first axis. The reference adapter calls the native forecast once per origin to ensure it averages exactly that origin's 32 assets, never another origin's observations.

The new evaluation-only block computes precisely the same reductions separately for each `[32,T,D]` group. It gathers the same top-three frequencies and computes the same per-asset softmax weights. At the coarsest length 14, integer periods can only be 14, 7, 4, 3 or 2. All origins selecting a given period are batched for the unchanged native time-imaging, dual-axis attention and multiscale-mixing operators. These operators do not mix distinct batch rows. If two frequencies give the same integer period, their identical branch outputs allow their weights to be added. The original residual addition and layer normalization are then applied. Every mixer block recomputes its own per-origin FFT selection.

Only the method dispatch is patched. During training it immediately invokes the original bound forward method. All module names, parameter shapes and state-dict keys remain unchanged, so original fitted checkpoints load strictly. During evaluation, grouped operations replace the Python origin loop while retaining independent-origin semantics. Floating-point summation order can differ slightly; mathematical equality does not imply bitwise equality.

## Verification

`tests/timemixerpp_fast_cpu_equivalence.json` compares 32 evenly spaced validation histories, four outer batch sizes 1/4/16/32, random initialization and weights after two genuine training updates. It verifies:

- Maximum normalized output error 2.98e−7 and maximum raw-return error 7.64e−9.
- Training loss and every parameter gradient agree exactly under matched RNG.
- State-dict names/shapes/count remain identical.
- All predictions are finite and no validation/test outcome is used for equality or timing.

The separate fast-runner CPU smoke test also passes real-data gradient, reload, ES and batch-independence checks. CPU timing is approximately unchanged; a CUDA timing test is needed to establish a speed improvement. Do not imply a GPU speedup before measuring it.

## CUDA checks and execution

Use the existing dependencies from the original point bundle. The optional checkpoint accepts the already fitted original `seed_1/candidate_01/best.pt`.

```bash
python test_timemixerpp_fast_eval.py --bundle . --data-bundle /path/official_supplement_bundle --out /path/fast_cuda_equivalence.json --device cuda --origins 32 --checkpoint /path/rebuttal/fresh_timemixerpp/seed_1/candidate_01/best.pt

python fresh_point_baselines_fast.py --family timemixerpp --repo /path/f-concord --bundle . --data-bundle /path/official_supplement_bundle --out /path/smoke_timemixerpp_fast --device cuda --smoke

python fresh_point_baselines_fast.py --family timemixerpp --repo /path/f-concord --bundle . --data-bundle /path/official_supplement_bundle --out /path/fresh_timemixerpp_fast --device cuda
```

The first command records initialization, two-update fitted weights, and the supplied real checkpoint in the same test. The equality thresholds are 2e−5 normalized and 1e−6 in raw return units; record actual discrepancies, not merely pass/fail. The frozen source manifest records original and optimized hashes. Default training remains five seeds, two learning rates, 100 full epochs and 15-check patience. No predictive metric or test outcome is used to choose the acceleration.
