# Full/FrozenGraph selected comparison

Entries are five-seed mean ± sample SD. MSE ×10⁴; ES ×10²; concept MAE in standardized units.

| Fold | Variant | MSE | ES | Concept MAE |
|---:|---|---:|---:|---:|
| 1 | Full | 6.8169 ± 0.0059 | 8.9770 ± 0.0144 | 0.4925 ± 0.0047 |
| 1 | FrozenGraph | 6.8185 ± 0.0031 | 8.9485 ± 0.0135 | 0.4916 ± 0.0011 |
| 2 | Full | 4.0787 ± 0.0025 | 7.6379 ± 0.0040 | 0.4700 ± 0.0187 |
| 2 | FrozenGraph | 4.0779 ± 0.0020 | 7.6342 ± 0.0045 | 0.4611 ± 0.0030 |
| 3 | Full | 3.3899 ± 0.0008 | 7.0094 ± 0.0052 | 0.4360 ± 0.0008 |
| 3 | FrozenGraph | 3.3902 ± 0.0001 | 7.0142 ± 0.0071 | 0.4327 ± 0.0007 |

All 30 final evaluations passed exact within-fold input/target alignment and artifact checks.
FrozenGraph retained A0 at every lead and exactly reproduced its graph-persistence diagnostics.
Paired differences and figure error bars describe matched training seeds. No result direction is assumed.
This selected control study is separate from the original forecasting and theory-checkpoint cohorts.
