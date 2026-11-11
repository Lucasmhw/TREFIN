# Matched-adapter descriptive calibration table

None; descriptive calibration table

| Method | Stage | MSE x10^4 | RMSE x10^2 | MAE x10^2 | ES x10^2 |
|---|---|---:|---:|---:|---:|
| TimeXer | before | 6.9771 ± 0.0215 | 2.6414 ± 0.0041 | 1.6955 ± 0.0034 | 9.3333 ± 0.0148 |
| TimeXer | after | 6.8370 ± 0.0010 | 2.6148 ± 0.0002 | 1.6720 ± 0.0001 | 9.2142 ± 0.0015 |
| TimeMixer++ | before | 6.9032 ± 0.0130 | 2.6274 ± 0.0025 | 1.6822 ± 0.0021 | 9.2723 ± 0.0121 |
| TimeMixer++ | after | 6.8363 ± 0.0005 | 2.6146 ± 0.0001 | 1.6719 ± 0.0001 | 9.2123 ± 0.0005 |

All rows use five seeds; means and sample SD are over seeds. Target hashes, origin alignment, saved per-origin loss means and checkpoint reload were verified. Source implementations are specified in each recorded protocol.
