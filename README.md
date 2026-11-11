# TREFIN

Code and experiment records for **TREFIN: Interpretable Financial Forecasting through Typed State Dynamics**. The implementation learns transitions of measured financial concepts and signed dependence graphs, then decodes those states into multivariate predictive distributions. The Python package retains its original `fconcord` name for compatibility with saved configurations and checkpoints.

## Repository and release contents

- `src/`, `configs/`, `runs/`, `rebuttal/`, `crossfold_v3/` and the baseline directories preserve the current execution snapshot. Configuration, selection, calibration, metric and log files are available directly in Git.
- `evidence/` contains selected additional scientific analyses, compact numerical arrays, figure data and generators. It includes the final normalized factor-flow certificate and the original reference source required by its recorded source hashes.
- Large server data, checkpoints and prediction arrays, together with historical server archives, are distributed in the same repository's **GitHub Releases**, under tag **`snapshot-2026-11-11`**, as **27 archive parts**. These parts belong to one snapshot; the release manifest and restoration utility provide the file lookup and integrity checks. A source-only checkout does not include these large server artifacts.
- [Experiment index](docs/EXPERIMENT_INDEX.md) identifies the completed cohorts, their evidence and reproduction entry points. [Bootstrap provenance](docs/BOOTSTRAP_PROVENANCE.json) records every local supplement and its SHA256. Existing server source and run files were not overwritten when these supplements were added.

Directory names such as `rebuttal`, `exploration` and `f-concord` are retained to preserve recorded paths and source provenance. Smoke, debug and partial runs remain distinguishable from the completed cohorts listed in the experiment index.

## Installation

Use Python 3.10 or later and a PyTorch installation suitable for your CPU or CUDA environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m fconcord --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. `pyproject.toml` specifies the core dependencies. The external baseline adapters additionally use their recorded upstream dependencies and source versions; consult their run environment records and licenses before reproducing those runs.

The following runs the original implementation tests and a small synthetic pipeline check. The synthetic check is not a financial benchmark:

```bash
python -m pytest tests
python -m fconcord run --config configs/debug.yaml --run-dir reproduction/synthetic_smoke
```

## Reproducing the financial protocols

Restore the release artifacts before evaluating saved checkpoints. For training from acquired inputs, the preparation script accepts the FinMultiTime time-series archive, its description CSV and the factor archive:

Download and verify the complete server artifacts with Python 3.12 or later:

```bash
python scripts/restore_artifacts.py
```

The [artifact guide](artifacts/README.md) explains selective restoration, storage requirements and checksums. Archives are hosted in the [complete experiment release](https://github.com/Lucasmhw/TREFIN/releases/tag/snapshot-2026-11-11).

```bash
python scripts/prepare_finmultitime.py --time-series-zip PATH_TO_TIME_SERIES_ZIP --description-csv PATH_TO_DESCRIPTION_CSV --factor-zip PATH_TO_FACTOR_ZIP --output-dir data/raw/finmultitime_us_32
python -m fconcord protocol --config configs/finmultitime_us_calibrated_protocol.yaml --output-dir reproduction/trefin_calibrated
```

The registered U.S. panel contains 32 assets with a five-day forecast horizon. The rolling tests cover 2020–2021, 2022–2023 and January 2024–March 2025, with seeds `1, 7, 19, 37, 73`. Prepared metadata, resolved configurations and per-run audit records define the exact samples and preprocessing. Training estimates the preprocessing parameters; validation selects configurations, checkpoints and calibration; the held-out test periods provide the reported evaluation.

The completed three-fold Full/FrozenGraph study has a separate isolated implementation and frozen selections. Its complete training orchestrator is:

```bash
python rebuttal/frozen_graph_v7/orchestrate_study.py --root . --study-root reproduction/frozen_graph_v7 --workers 6
```

Read the study's registration and the [experiment index](docs/EXPERIMENT_INDEX.md) before launching it. A complete rerun is a substantial training workload. The study compares learned graph transitions with retaining the same initial measured graph, using matched training, configuration selection and calibration.

## Checking compact evidence without retraining

The final factor-flow verification recomputes all saved coefficient compositions and aggregate statistics from the compact arrays. It uses the frozen original reference source, not the subsequently extended training implementation:

```bash
python evidence/theory_v3/exploration/verify_flow_compact.py --theory-root evidence/theory_v3 --model-source evidence/reference_source/src/fconcord/model.py --output flow_verification.json
```

Regenerate the concept persistence, graph persistence and direct mean-response figures from saved measurements:

```bash
python evidence/scripts/plot_rebuttal_evidence.py --evidence-dir evidence/analyses/archived_evidence --sensitivity-dir evidence/analyses/mean_sensitivity --metadata compact_prepared/fold_1.json --output-dir reproduction/figures
```

See the experiment index for the interval calibration, structural, matched-baseline and crossfold figure inputs. The numerical certificates describe local sensitivity of the learned transition and decoder on specified state neighborhoods; they are not a bound on forecasting error against an unknown true conditional law.

## License

The original implementation is released under the [MIT License](LICENSE). Vendored external implementations retain their own licenses and attribution; see [third-party notices](THIRD_PARTY_NOTICES.md). Data and pretrained upstream assets remain subject to their respective providers' terms; see [data provenance](DATA_PROVENANCE.md).
