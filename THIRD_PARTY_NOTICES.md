# Third-party notices

The TREFIN research workspace includes first-party implementation and experiment scripts alongside selected upstream implementations. The project code license does not replace the licenses of these upstream components, their dependencies, or the underlying datasets. Original copyright headers and license texts are retained with the corresponding files.

## Vendored implementations

| Component | Pinned upstream revision | License | Representative bundled license |
| --- | --- | --- | --- |
| [TEDM](https://gitlab.com/dlr-dw/tedm) | `1d65888cc1ead587f738a0081a150d3c15d97b17` | CC BY-NC-SA 4.0; copyright 2026 TEDM Authors | [official/upstream/tedm/LICENSE](official/upstream/tedm/LICENSE) |
| [OpenLTM / Timer-XL](https://github.com/thuml/OpenLTM) | `0b3005099d380ecc00d512f72553c7fe8fccca99` | MIT; copyright 2022 THUML @ Tsinghua University | [official/upstream/OpenLTM/LICENSE](official/upstream/OpenLTM/LICENSE) |
| [Time-Series-Library / TimeXer](https://github.com/thuml/Time-Series-Library) | `4e938a1767106324dd753b2a44832bf870a0252e` | MIT; copyright 2021 THUML @ Tsinghua University | [official_point/fresh_point_bundle/upstream/TimeSeriesLibrary/LICENSE](official_point/fresh_point_bundle/upstream/TimeSeriesLibrary/LICENSE) |
| [PyPOTS / TimeMixer++](https://github.com/WenjieDu/PyPOTS) | `2271533587bab250163437ed39cfe13cefd8b80a` | BSD-3-Clause; copyright 2023-present Wenjie Du | [official_point/fresh_point_bundle/upstream/PyPOTS/LICENSE](official_point/fresh_point_bundle/upstream/PyPOTS/LICENSE) |

The experiment bundles preserve additional copies of these components under `official_point_fast/fresh_point_fast_bundle/upstream/`, `crossfold_v3/point_bundle/upstream/`, and `crossfold_v3/data/fold_2/upstream/` and `fold_3/upstream/`. Each copy retains its applicable license. The same component-level terms apply to copies contained in archived experiment bundles.

TEDM is distributed under its original **Attribution-NonCommercial-ShareAlike 4.0 International** license, not the first-party code license. Uses and adaptations of that component remain subject to its attribution, noncommercial, and share-alike conditions. The included MIT and BSD components retain their respective notices and conditions.

## Implementation provenance

The pinned revisions and per-file SHA-256 digests are recorded in [official/manifest.json](official/manifest.json), [official_point/fresh_point_bundle/manifest.json](official_point/fresh_point_bundle/manifest.json), and the corresponding cross-fold manifests. Local experiment runners, preparation scripts, and calibration code are stored alongside the upstream modules. Consult those manifests and scripts to distinguish the upstream implementation from the experiment-specific integration.

Python packages installed to run the experiments retain their own licenses; this notice does not relicense those dependencies. Model checkpoints and other derived artifacts should be used with the applicable implementation and source-data terms. Dataset provenance and attribution are documented separately in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).
