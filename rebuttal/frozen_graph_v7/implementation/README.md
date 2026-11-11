# FrozenGraph v7 implementation

This isolated study compares newly fitted Full and FrozenGraph models across the three original chronological folds and seeds 1, 7, 19, 37, and 73. It does not replace or alter the original results or their checkpoints.

## Controlled change

FrozenGraph uses the augmented transition `(C,A) -> (F_C(C,A),A)`. Thus every future graph is exactly the observed initial graph `A0`, including the graph passed to the concept transition and return decoder. Typed propagation, local `A C` features, signed graph precision, market-state inputs, the decoder and all active trainable components remain available. This control addresses the incremental value of learned graph evolution relative to a retained, static graph channel; NoGraph instead removes that channel.

The graph-target coefficient is zero because the frozen path has no trainable dependence. The actual augmented-state gain bound is `[[m_CC,m_CA],[0,1]]`, with radius `max(m_CC,1)`. The original gain coefficient 0.1 and target 0.995 are retained. The identity graph row yields a parameter-independent minimum penalty of 0.0000025; gradients can affect this penalty when `m_CC > 1`. The retained `m_CA` accounts for sensitivity to the input graph. Learned proposal, refresh and gate terms do not enter this bound. `torch.maximum` gives finite autograd behavior at the tie. These statements define the intervention and objective; they are not explanations of any performance outcome.

Unused proposal encoders remain allocated, preserving identical initialization order and state-dictionary layout. They are not evaluated and receive no gradient under FrozenGraph. All active components train from initialization. The Full execution is unchanged; CPU and CUDA semantic fixtures verify bitwise forward/loss/gradient/optimizer equivalence to the registered baseline model.

## Selection and evaluation

Each fold independently searches seed 1 over the original four nominal candidates, in order: base, graph refresh 0.02, residual scale 0.05 with refresh 0.10, and concept/graph coefficients 0.10/0.02. FrozenGraph always overrides the graph coefficient to zero. Its refresh-only candidate is identical to base, is recorded as an alias and is not refitted or replaced. Consequently the search contains 12 Full fits and 9 FrozenGraph fits.

Every unique fit uses AdamW with the original learning rate 0.001 and weight decay 0.0001, batch size 64, 20-epoch ceiling, patience 6, gradient clip 1, 16 training scenarios and 64 validation scenarios. The unchanged trainer selects the checkpoint using uncalibrated validation ES (minimum improvement 1e-6), then fits the positive mean scalar rho within [0.05,1] using validation data alone, tolerance 1e-5. The candidate with minimum calibrated validation ES wins; exact ties follow the registered order.

Search and final training physically exclude test-origin rows from the object provided to the trainer. `freeze_selections.py` verifies all six fold/variant choices and their checkpoint hashes before writing `frozen_selections.json`. Final evaluation is blocked until this global freeze exists. The selected seed-1 checkpoint is reused before its first test evaluation; the four other seeds are fitted with the same selected architecture and training settings. This gives 30 final evaluations from 21 search fits plus 24 additional fits. Test evaluation uses 256 scenarios, the original targets and chronological test origins. No test outcome controls model, checkpoint, calibration, or candidate selection.

## Invocation

Install `model.py` as `STUDY/src/fconcord/model.py` within an isolated copy of the original source. Set `PYTHONPATH=STUDY/src`. Retain these scripts together under `STUDY/implementation`. `REPO` is the original repository containing all 15 resolved configuration files. `PREPARED/fold_1`, `fold_2`, and `fold_3` each contain the exact restored `prepared_arrays.npz` and `prepared_metadata.json`.

Run the six independent validation searches (one per fold/variant), then freeze once, then run all 30 selected evaluations:

```sh
python implementation/preflight.py --root REPO --prepared-root PREPARED --out preflight_gpu.json --device cuda --threads 2
python implementation/search_validation.py --root REPO --prepared-root PREPARED --study-root STUDY --fold 1 --variant Full --device cuda --threads 2
# Repeat the search command for every fold in 1,2,3 and variant in Full,FrozenGraph.
python implementation/freeze_selections.py --study-root STUDY
python implementation/run_selected.py --root REPO --prepared-root PREPARED --study-root STUDY --fold 1 --variant Full --seed 1 --device cuda --threads 2
# Repeat evaluation for all three folds, both variants and five registered seeds.
```

The parent orchestration script registers the overall study and schedules these commands. The preflight takes one optimizer step on training data solely to check shapes, finite outputs/gradients, invariants and configuration compatibility. It saves no study checkpoint and makes no selection; every real fit resets its seed independently.

## Artifacts and definitions

Each `runs/fold_F_seed_S/Variant` contains the complete training configuration, calibration summary, checkpoints, full validation/test predictions, original calibrated and uncalibrated metric suites, and a completion record with hashes. The uncalibrated suite reverses the scalar mean translation using the same innovations. The original evaluator is unchanged. Float64 per-origin MSE, MAE and ES are additionally saved for paired comparisons; the finite-sample ES uses every unordered scenario pair divided by `S(S-1)` and averages the five lead-specific cross-sectional scores. Original float32 aggregate reductions can differ from these double-precision reductions by rounding.

`tables/states.json` excludes lead 0 when reporting errors, retains all five leads and 26 concept coordinates, and reports learned and persistence graph errors. Graph support is `abs(A_ij)>0.05` off the diagonal, with empty-union IoU zero. FrozenGraph's maximum absolute departure from `A0` is checked to be exactly zero. `tables/origin_losses.npz` contains indices, raw origins, targets, means and double-precision paired losses. `audit/provenance.json` records source/input/selection hashes; `audit/test_evaluation_started.json` records the frozen choice and selected checkpoint before test collection. Incomplete outputs are preserved and never silently overwritten.

Semantic fixtures and preflight diagnostics are implementation validation, not forecasting evidence. Only completed registered runs contribute experimental results.
