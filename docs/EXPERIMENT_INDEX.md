# Experiment and evidence index

This index distinguishes completed scientific cohorts from the smoke, partial, diagnostic and historical records retained in the execution snapshot. Paths are relative to the repository root. Large checkpoint and server-array paths become available after restoring the 27-part `snapshot-2026-11-11` release. Compact supplemental arrays under `evidence/` are included in Git.

## Completed cohorts

| Family | Completed scope and protocol | Records and implementation |
|---|---|---|
| Original U.S. TREFIN | Three folds × five seeds. Calibrated and uncalibrated stages are separate recorded evaluations. | `runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1/`; `runs/protocols/finmultitime_us_rolling_3fold_5seed_v1/`; `configs/finmultitime_us_*protocol.yaml`; `src/fconcord/` |
| Four matched external backbones | TEDM, Timer-XL, TimeXer and TimeMixer++; 60 selected model/fold/seed runs over three folds and five seeds. Configuration selection and mean calibration use validation only. Preserve the adapter and likelihood/scenario conventions in the resolved protocols. | Fold 1: `rebuttal/fresh_tedm_complete/`, `fresh_timerxl/`, `fresh_timexer/`, `fresh_timemixerpp_complete/`. Folds 2–3: `rebuttal/crossfold_v3/`; source and execution plan: `crossfold_v3/`; consolidated results: `evidence/analyses/crossfold_v3/` |
| Nine native matched-calibration models | Nine implementations × five seeds on fold 1 = 45 matched records. A historical directory name containing `3fold` does not establish completion of three folds. | `evidence/analyses/native_calibration_v2/`; `rebuttal_scripts/native_matched_calibration.py`; original records under `runs/benchmarks/finmultitime_us_native_3fold_5seed_v1/` |
| Five structural variants | Full, NoGraph, signed-only, unsigned-only and no-state-supervision; five seeds on fold 1. Fixed-setting and validation-selected protocols have 25 final records each. | `rebuttal/retraining/`, `rebuttal/retraining_tuned/`, `rebuttal/search_A/`, `rebuttal/search_B/`; `rebuttal_scripts/run_retraining.py`, `search_validation.py` and `summarize_retraining.py` |
| Full versus FrozenGraph | Three folds × two variants × five seeds = 30 final evaluations. Twenty-one unique search fits and 24 additional selected fits; six fold/variant selections frozen before test evaluation. FrozenGraph retains the measured initial graph throughout the rollout. | `rebuttal/frozen_graph_v7/`; `frozen_selections.json`, `orchestration_registration.json`, `input_equivalence.json`, `src/`, `analysis/` and `analysis_tools/` within that directory |
| Financial references | Zero, training-mean and zero-mean EWMA-Student comparisons across the three folds, aligned with the 15 original TREFIN reference checkpoints. | `evidence/analyses/archived_evidence/`, `ewma_student/`, `ewma_comparison/`; `rebuttal_scripts/ewma_student_baseline.py` and `compare_ewma_energy.py` |
| Paired temporal comparisons | Seed-averaged origin-level losses; registered blocks 5/10/20 and DM HAC lag 20. The two requested matched backbones contribute 18 primary model/fold/loss rows. Keep each comparator's recorded cohort separate. | `evidence/analyses/crossfold_v3/paired_tests.csv`, `paired_seed_average_differences.npz`; `evidence/analyses/archived_evidence/paired_tests.csv`; `evidence/crossfold_v3/analysis/` |
| Measured-state persistence | All 15 original reference checkpoints; five leads per fold; all 26 concept coordinates. Continuous state/graph errors and thresholded graph IoU are distinct measurements. | `evidence/analyses/archived_evidence/concept_persistence.csv`, `graph_persistence.csv`, `evidence.json`; `rebuttal_scripts/evaluate_archived_evidence.py` |
| Innovation scale calibration | All 15 reference checkpoints. Original reference, fixed validation 90% coverage scaling, fixed validation ES scaling and causal rolling 90% scaling are distinct protocols. | `evidence/analyses/interval_calibration/`, `rolling_interval_calibration/`; `rebuttal_scripts/calibrate_intervals.py`; `evidence/scripts/rolling_interval_calibration.py` |
| Direct mean sensitivity | Fifteen checkpoints; 32 origins per checkpoint; five leads, 32 assets and all 26 concepts. There are 25,600 derivatives per concept and fold. | `evidence/analyses/mean_sensitivity/`; `rebuttal_scripts/mean_sensitivity.py`; plotted values under `evidence/figures/` |
| Local quantitative certificates | Fifteen checkpoints and 480 trajectories. Four concept/graph radius pairs: (0.001,0.0005), (0.01,0.005), (0.1,0.05), (1,0.5). There are 1,920 trajectory/radius cases and 9,600 decoder neighborhoods. | `evidence/theory_v3/` and its `exploration/` subdirectory; details below |
| Original auxiliary experiments | Original single-seed pilot ablations, checkpoint interventions, hyperparameter sensitivity, runtime and the recorded single-seed BTCF transfer experiment retain their original scopes. | Original `runs/` records and resolved configurations. Do not pool these with the later matched multiseed cohorts. |

Historical external-model aggregate summaries are retained as historical summaries. They are distinct from the reproducible matched task-adapter cohorts above; missing historical weights or predictions must not be inferred from an aggregate table.

## Final theory implementation and provenance

The state-bound implementation is `evidence/theory_v3/tube_bounds.py`, with `local_mlp_bounds.py` and `local_tube_results_v2/`. `transition_defect_results/` records admissible nonzero transition-defect budgets. `decoder_bounds.py` and `decoder_results/` retain the original full-decoder bound.

The final dependence coupling is `evidence/theory_v3/exploration/normalized_factor_flow.py`. Its derivation, `all_radius_flow_results/`, `validation/` and portable reproduction scripts use the same checkpoints, state neighborhoods and other decoder terms. The final coefficient is the componentwise minimum of the original and normalized factor-flow dependence coefficients. At concept radius 0.1 and graph radius 0.05, the foldwise median local decoder coefficients are 36.084, 43.536 and 31.318. Smaller and larger radii remain in the saved complete results.

The exact original implementation used for the certificate is isolated under `evidence/reference_source/src/`; its `model.py` SHA256 is `097ac9b51db52eda88fd661566cf36b99e60cc9ea956ed7739975cbcc5ab3eee`. Current training source remains under `src/`; the FrozenGraph study also preserves its own source. These implementations must not be interchanged when checking recorded hashes.

The saved `new_W1` and `old_W1` fields propagate initial error `0.1R` with zero additive transition defects. Nonzero-defect budgets are separate certified inputs to the same state recursion. Analytic inequalities establish the local uniform bounds; finite checkpoint and matrix checks test their implementation. Neither operation estimates an error to the unknown true forecasting law.

After restoring the original calibrated checkpoint/prepared-input cohort, the final coefficients can be recomputed with:

```bash
python evidence/theory_v3/exploration/reproduce_factor_flow.py --all-radii --source evidence/reference_source/src --protocol runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1 --state-results evidence/theory_v3/local_tube_results_v2 --decoder-results evidence/theory_v3/decoder_results --output reproduction/factor_flow
python evidence/theory_v3/exploration/reproduce_factor_flow_validation.py --source evidence/reference_source/src --protocol runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1 --state-results evidence/theory_v3/local_tube_results_v2 --output reproduction/factor_flow_validation
```

Use `--folds 1 --seeds 1` for a smaller reproduction check. The complete scientific result uses all three folds and all five registered seeds. The compact verification command in the root README needs no checkpoints or test labels.

## Figures and summaries

`evidence/figures/` contains the scientific figures and their plot data, captions and provenance. The generator commands below write new outputs instead of modifying the recorded evidence:

```bash
python evidence/scripts/plot_interval_calibration.py --input evidence/analyses/rolling_interval_calibration/aggregate.csv --output reproduction/interval_figures
python evidence/scripts/plot_rebuttal_evidence.py --evidence-dir evidence/analyses/archived_evidence --sensitivity-dir evidence/analyses/mean_sensitivity --metadata compact_prepared/fold_1.json --output-dir reproduction/state_figures
```

Other generators are `evidence/scripts/plot_structural_comparison.py`, `plot_native_calibration.py`, `evidence/crossfold_v3/plot_crossfold_comparison.py` and `evidence/theory_v3/plot_theory_certificate.py`. Their argument parsers define the input summary/manifest paths. Some original analysis scripts preserve execution-machine defaults; pass their explicit path arguments or adjust a separate working copy rather than overwriting recorded source and hashes.

## Reading run records

For each comparison, use the resolved configuration, selected checkpoint, calibration record and test metrics belonging to the same fold/seed. Keep before/after calibration metrics together, preserve candidate-selection decisions, and align per-origin losses before aggregation. Prepared metadata and source manifests document the measured state definitions, normalization and sample boundaries. The release preserves large arrays needed to recompute scores and inspect checkpoints.

`docs/BOOTSTRAP_PROVENANCE.json` lists each supplementary file and its original archive location, byte size and SHA256, including files retained from the server rather than overwritten. Internal reviewer-response exports, manuscript drafts and private connection helpers are outside this scientific supplement.
