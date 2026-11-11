# F-CONCORD task continuation record — 2026-08-21

## Current priority: calibrated F-CONCORD before baselines

At the user's request, all baseline/foundation/ablation jobs were stopped cleanly. Six of
fifteen native-baseline units and partial official-model caches are preserved for later
recovery; no baseline process is active.

The original F-CONCORD protocol remains complete at 15/15. A validation-only predictable-
signal gate was added for a separate calibrated protocol. It preserves scenario innovations
and dependence and minimizes a bounded convex Energy Score objective over `rho in [0.05,1]`.
The implementation is in `src/fconcord/calibration.py`; `rho` is saved in both the resolved
config and `best.pt` before fresh-model reload and the one-time final test. Server Ruff and
the full 63-test suite pass. The real-data two-epoch fold-1 debug selected `rho=0.543579`,
reduced validation Energy Score from `0.06486499` to `0.06482850`, reloaded with zero
prediction difference, completed one final test, generated 50 figures, and passed run audit.
It is implementation evidence, not a paper result.

The first formal launch was stopped before completion because its run-local methodology
snapshot predated the new gate. That entire attempt was retained as
`finmultitime_us_calibrated_rolling_3fold_5seed_v1_precontract_aborted_20260821T1936`.
After updating `METHODOLOGY_CODE_SPEC.md`, `manual_contract.yaml`, `MODEL_CODE_MAP.md`, and
the README locally/server-side, the clean formal protocol was restarted at
`/root/autodl-tmp/f-concord/runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1`.
Only this clean directory may be aggregated.

## Continuation update: equal-budget comparisons

Execution boundary is explicit: the local notebook stores/edits the repository and compact
evidence only. All tests, debug runs, training and evaluation run on the server at
`/root/autodl-tmp/f-concord`.

Implemented and saved locally/server-side:

- equal-budget candidate selection with validation-only selection and one test access;
- ShrunkVAR, shared HistGradientBoosting, StateMLP, RawHistoryMLP, DLinear, GRU, PatchTST,
  iTransformer, MASTER-style and TimeMixer-style registered baselines;
- official zero-shot adapters for `amazon/chronos-2` and
  `google/timesfm-2.5-200m-pytorch`;
- true retraining variants `NoGraph`, `NoMarketState`, `NoRiskConcepts`,
  `NoForwardConceptSupervision` and `GaussianSymmetricDecoder`;
- DM--HAC, circular-block bootstrap, Holm, block-bootstrap SPA and MCS;
- independent benchmark/ablation audits and seed-averaged origin-level comparison builder.

Server verification: 59 tests passed and Ruff passed after the phase-3 sync. CLI import and
targeted Ruff checks passed after the audit/comparison additions. Official packages are
isolated in `.foundation-venv`: `chronos-forecasting==2.3.1`, `timesfm==2.0.2`, and
`transformers==5.15.1`; the main `.venv` was not modified.

Real-data debug evidence:

- native 9-model matrix: 9/9 models, 18/18 candidates, nine one-time tests, zero maximum
  checkpoint-reload difference, completed in about 61 seconds with two debug epochs;
- fully retrained `NoGraph`: validation selected epoch 4, early stopped at epoch 10,
  checkpoint reload difference zero, test RMSE 0.02617147 and Energy Score 0.09055707.
  This debug preceded the formal `lambda_graph=0` cleanup and is not a paper result.

Current server state after the user's priority change:

- all baseline training, foundation downloading, and the remaining-protocol queue are
  stopped; no F-CONCORD or baseline process is using the GPU;
- native baseline output is preserved at
  `/root/autodl-tmp/f-concord/runs/benchmarks/finmultitime_us_native_3fold_5seed_v1`;
  6 of 15 fold/seed units completed before the clean stop and can be recovered later;
- official foundation caches are partially/fully preserved and remain resumable;
- the current priority is to inspect and improve F-CONCORD itself using train/validation
  evidence only. Do not resume baselines unless the user requests it.

The stopped queue had been updated to run TimeMixer, official foundations, 75 retrained
ablations, independent audits, final comparison, and comparison figures. Its latest log is
`/root/autodl-tmp/f-concord/runs/remaining_protocols_driver_v3.log`; this is historical state,
not an active job.

This file is the restart point for the current Codex task. Do not repeat completed runs.

## Completed and verified

- The code-normative F-CONCORD repository is implemented locally and mirrored at
  `/root/autodl-tmp/f-concord` on the server.
- The official FinTSBridge BTCF H=5 audit and the FinMultiTime US-32 single-run audit are
  complete; identities and hashes are in `docs/REAL_RUNS.yaml`.
- The registered FinMultiTime US-32 H=5 protocol completed all 3 folds x 5 seeds
  `[1, 7, 19, 37, 73]` (15/15 runs).
- Full server protocol:
  `/root/autodl-tmp/f-concord/runs/protocols/finmultitime_us_rolling_3fold_5seed_v1`
- Server compact archive:
  `/root/autodl-tmp/finmultitime_us_rolling_3fold_5seed_v1_compact.tar.gz`
- Local compact archive and extracted evidence:
  `server_evidence/finmultitime_us_rolling_3fold_5seed_v1_compact.tar.gz` and
  `server_evidence/finmultitime_us_rolling_3fold_5seed_v1/`.
- Compact archive SHA-256:
  `F17D7344797AD69632800F6B6E45766C65919E0E9269E02ED5549B0429EDF2BA`.
- Independent full server audit: 15 runs, 30 loadable checkpoints, zero reload difference,
  one final-test sequence per run, all label paths within preregistered split ends, 750
  figures, and all per-run artifact audits passed.
- Independent local compact audit passed and is saved as
  `protocol_audit_local_compact.json`; the updated server-full report was copied as
  `protocol_audit_server_full.json`.
- Local and server test suites each pass 31/31; Ruff passes on both systems.

## Headline evidence (do not overclaim)

- Equal-run mean ± sample standard deviation across 15 runs:
  RMSE `0.02160468 ± 0.00343585`, MAE `0.01449907 ± 0.00184476`, directional accuracy
  `0.51142840 ± 0.00608571`, Energy Score `0.07888278 ± 0.00862062`, msIC
  `0.00194939 ± 0.00405917`, msIR `0.00399293 ± 0.00818762`.
- ZeroReturn has slightly lower mean RMSE/MAE (`0.02158286/0.01447850`), so there is no
  predictive-SOTA claim.
- All sampled local Jacobian diagnostics are below one (maximum `0.97396153`), but every
  conservative global structural certificate fails (radius range `43.36068–60.65117`).
  Never substitute the local diagnostic for the failed global theorem premise.
- E04 is now implemented. E01, E02, E05 and E06 remain partial; E03 remains pending. The
  complete three-layer ICLR study is not finished.

## Files changed at the pause point

- `README.md`: full reproduction instructions, audited real runs and protocol results.
- `docs/METHODOLOGY_CODE_SPEC.md`: code-normative methodology and frozen adaptations.
- `docs/manual_contract.yaml`: requirement-by-requirement status; E04 updated to implemented.
- `docs/REAL_RUNS.yaml`: run IDs, hashes, metrics and claim limits.
- `scripts/audit_protocol.py`: portable full/compact protocol audit and comparison aggregation.
- The remainder of the implementation and tests are visible in `git status`; the worktree is
  intentionally not reset or cleaned.

## Resume checklist

1. Read this file, `README.md`, `docs/METHODOLOGY_CODE_SPEC.md`,
   `docs/manual_contract.yaml`, and `docs/REAL_RUNS.yaml` before changing anything.
2. Do not run tests, training, or evaluation on the local notebook. It is only for source
   editing, storage, and compact evidence.
3. Run all tests and Ruff checks on the server, and do not rerun the completed 15 F-CONCORD
   protocol jobs unless evidence is corrupt or a behavior-changing core-model revision is
   intentionally introduced.
4. Continue the unfinished publishable matrix in priority order: equal-budget modern
   baselines and retrained ablations/statistical tests (E05); formal calibration/tail/economic
   metrics (E06); remaining FinTSBridge horizons/tasks (E02); TLOB external validation (E03);
   and point-in-time/delisting/China equity evidence (E01).
5. Keep executed code + resolved config + run audit normative. Update the contract and README
   before presenting any new result, and preserve negative findings.

No SSH password or other credential is stored in this record or anywhere in the repository.

## Continuation update: validation-calibrated protocol

- Completed `finmultitime_us_calibrated_rolling_3fold_5seed_v1`: 3 rolling folds x seeds
  `[1, 7, 19, 37, 73]`, 15/15 runs on the server.
- Independent server audit passed: 30 loadable checkpoints, zero fresh-reload differences,
  one final test per run, temporal purges and label boundaries verified, consistent source
  and implementation hashes, and 750 figures.
- Equal-run test means are MSE `0.0004761721`, RMSE `0.02157191`, MAE `0.01447353`,
  directional accuracy `0.51142840`, Energy Score `0.07873894`, msIC `0.00194939`, and
  msIR `0.00399293`.
- Relative to ZeroReturn MSE `0.0004767555`, equal-run mean out-of-sample R2 is
  `0.00081775`; fold-level values are `0.00276428`, `-0.00049655`, and `0.00018551`.
  The advantage is very small and not consistent across folds.
- MSE is now a native train/validation/test metric in future runs. A deterministic report
  derived from the completed immutable test archives is saved as
  `protocol_squared_error_metrics.json`; MAPE is intentionally excluded.
- Validation signal calibration passed for all 15 runs, with rho in `[0.05, 0.71794042]`,
  no validation Energy-Score worsening, and maximum pairwise-invariance error `5.56e-17`.
- The full server suite passes 66 tests and Ruff. No tests or experiments were run on the
  local notebook.
- The exploratory calibrated protocol is not confirmatory evidence because its mechanism
  was developed after viewing the earlier protocol's aggregate test behavior. Preserve this
  limitation in any paper draft.
