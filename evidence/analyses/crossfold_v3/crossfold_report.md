# Completed three-fold baseline comparison

All four baseline families have five completed seeds in each fold. The two new folds use the frozen fold-1 fitting protocol. The table reports mean ± sample SD over seeds after validation-only mean calibration. Full before/after eight-metric values are in metrics.csv.

| Fold | Method | MSE ×10⁴ | MAE ×10² | ES ×10² | 90% coverage |
|---|---|---:|---:|---:|---:|
| 1 | TEDM | 6.8296 ± 0.0003 | 1.6743 ± 0.0000 | 10.5919 ± 0.0005 | 30.3060 ± 0.0251 |
| 1 | Timer-XL | 6.8352 ± 0.0008 | 1.6718 ± 0.0001 | 9.2117 ± 0.0026 | 77.4040 ± 0.1784 |
| 1 | TimeXer | 6.8370 ± 0.0010 | 1.6720 ± 0.0001 | 9.2142 ± 0.0015 | 77.4822 ± 0.1149 |
| 1 | TimeMixer++ | 6.8363 ± 0.0005 | 1.6719 ± 0.0001 | 9.2123 ± 0.0005 | 77.5935 ± 0.0271 |
| 1 | TREFIN | 6.8171 ± 0.0066 | 1.6704 ± 0.0003 | 8.9744 ± 0.0160 | 80.1230 ± 0.2885 |
| 2 | TEDM | 4.0795 ± 0.0000 | 1.4305 ± 0.0000 | 9.1533 ± 0.0011 | 30.1570 ± 0.0432 |
| 2 | Timer-XL | 4.0761 ± 0.0008 | 1.4291 ± 0.0002 | 7.7254 ± 0.0011 | 81.9677 ± 0.0241 |
| 2 | TimeXer | 4.0767 ± 0.0003 | 1.4292 ± 0.0001 | 7.7278 ± 0.0008 | 81.9471 ± 0.1037 |
| 2 | TimeMixer++ | 4.0767 ± 0.0002 | 1.4291 ± 0.0003 | 7.7263 ± 0.0003 | 82.0754 ± 0.0804 |
| 2 | TREFIN | 4.0781 ± 0.0022 | 1.4299 ± 0.0004 | 7.6372 ± 0.0028 | 85.7949 ± 0.8491 |
| 3 | TEDM | 3.4004 ± 0.0000 | 1.2448 ± 0.0000 | 8.4141 ± 0.0005 | 30.4820 ± 0.0303 |
| 3 | Timer-XL | 3.3905 ± 0.0013 | 1.2415 ± 0.0003 | 7.0099 ± 0.0021 | 88.7484 ± 0.2928 |
| 3 | TimeXer | 3.3903 ± 0.0002 | 1.2417 ± 0.0001 | 7.0100 ± 0.0003 | 88.7492 ± 0.1399 |
| 3 | TimeMixer++ | 3.3905 ± 0.0002 | 1.2418 ± 0.0001 | 7.0096 ± 0.0001 | 88.9849 ± 0.0297 |
| 3 | TREFIN | 3.3899 ± 0.0007 | 1.2417 ± 0.0003 | 7.0101 ± 0.0054 | 86.5396 ± 0.3557 |

Paired inference is performed separately within each fold after seed averaging. Holm adjustment covers the six TEDM/Timer-XL × fold comparisons separately for each loss and block choice. The original fold-1 two-comparator adjusted outputs are preserved separately; per-comparison intervals are not simultaneous intervals. TimeXer and TimeMixer++ are descriptive comparisons.
