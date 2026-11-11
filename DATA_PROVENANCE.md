# Data provenance and attribution

This release preserves the scientific inputs, fitted artifacts, and experimental records used by the accompanying TREFIN code. Source-code licensing and source-data terms are separate: the project code license does not relicense third-party financial observations. The original sources and transformations are described below.

## US equity panel and daily factors

The equity preparation code is [src/fconcord/data/finmultitime.py](src/fconcord/data/finmultitime.py). It builds a fixed 32-asset US cohort from the time-series and stock-description files distributed with **FinMultiTime**, using training-era liquidity for selection and a common calendar for the panel and factors. It preserves source-file SHA-256 digests in the preparation metadata. The comparison manifests record asset order, 126-day histories, five-day horizons, fold boundaries, normalization rules, and data digests; see [official/manifest.json](official/manifest.json) and [crossfold_v3/data/fold_2/manifest.json](crossfold_v3/data/fold_2/manifest.json), with the corresponding fold 3 manifest.

- **FinMultiTime:** [author dataset repository](https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting) and [dataset paper](https://arxiv.org/abs/2506.05019). The author-provided dataset card declares an **MIT** license. The paper identifies Yahoo Finance as the source of its US price and volume observations. Attribution to the dataset creators and the original financial-data source is retained; the dataset-card declaration is distinct from the license on TREFIN code.
- **Daily five-factor data and risk-free rate:** [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), attributed to **Eugene F. Fama and Kenneth R. French**. The adapter parses the official daily archive, converts percentage simple returns to decimal returns, and converts the risk-free rate with `log1p` for subtraction from log asset returns. The code license does not grant new rights over these factor observations; attribution and the applicable source terms remain with the original data.

Prepared return and target arrays are derived financial data, not merely model scores. Per-fold manifests preserve the panel and factor digests and, where recorded, the exact origin, target, and split-array digests. These records identify the data vintage used by the experiments rather than substituting a later download of an evolving public dataset.

## FinTSBridge BTCF

The secondary benchmark uses the **BTCF** artifact distributed by FinTSBridge. The adapter [src/fconcord/data/fintsbridge.py](src/fconcord/data/fintsbridge.py) identifies:

- [FinTSBridge repository](https://anonymous.4open.science/r/FinTSBridge-0D6C/), repository-file identifier `fb187ff4`;
- artifact path `FinTSBridge_models/TSLib_baseline/dataset/FBD/BTCF.csv`;
- an expected 43,014 hourly observations, with SHA-256 computed for the actual downloaded bytes.

The adapter reconstructs positive relative spot and futures prices from the benchmark's normalized log-price paths, reconstructs volume from log-volume, and derives the order-flow inputs from the supplied total and taker-buy volumes. The relative price scale is documented in its preparation metadata. See the [FinTSBridge paper](https://arxiv.org/abs/2503.06928) for the benchmark and its Binance source attribution.

BTCF remains an upstream data artifact: the FinTSBridge paper's publication license is not treated as a separate license for its CSV data, and TREFIN's code license does not relicense that CSV. For new downloads of Binance public data, consult the [official Binance public-data terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md). Their version 1.0, dated 26 August 2026, specifies CC BY-NC-SA 4.0 and prospective application; it should not be assumed to determine the terms of an earlier download. Preserve the original attribution, data vintage, and applicable source terms when reusing or redistributing these data.

## Experimental artifacts

The release also contains predictions, per-origin losses, calibration measurements, fitted concept responses, checkpoints, configuration histories, and aggregate result tables. They serve different reproducibility purposes: observations and targets identify the evaluation data; predictions and losses support reported comparisons; fitted parameters and configurations support rerunning the methods. Large binary artifacts and archived bundles are distributed with the release's file inventory and SHA-256 checksums. Use the supplied manifests to verify exact files and retain the component and source-data notices when redistributing derived bundles.

Upstream implementation versions and licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
