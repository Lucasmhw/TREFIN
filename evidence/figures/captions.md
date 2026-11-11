## R1_concept_persistence

Revised concept-path diagnostic. (a) Fold-1 coordinate-level MAE skill relative to persistence, 100(1 − MAE_learned/MAE_persistence), for all 26 concepts and leads 1–5. Each MAE is averaged over the same 500 held-out origins, 32 assets and five fitted seeds. Positive values indicate lower learned MAE; negative values favor persistence. Colors use a symmetric linear scale centered at zero without clipping; cell labels are rounded percentages, and exact values are in the source table. (b–d) Absolute mean concept MAE across the three chronological folds, in train-fitted robust-standardized projected coordinates. Persistence carries the measured input C_0 to every future lead. Curves are five-seed means.

## R2_graph_persistence

Revised graph-path diagnostic. Mean per-origin support intersection-over-union for learned graph forecasts and matched persistence A_hat_h=A_0, using the same targets and origins in each fold. Support is defined by strict |A_ij|>0.05 on off-diagonal entries, and empty unions receive zero, exactly matching the stored-evidence evaluator. Lines average over five seeds; blue bands show ±1 sample standard deviation across seeds. Persistence is deterministic for the fixed prepared data. Measured graphs are carried forward between five-observation refits. Persistence has higher IoU at every displayed lead in all three folds. The full IoU range is shown.

## R3_direct_mean_signed

Direct own-asset mean sensitivities at evolved predicted states. For each fold, derivatives ∂μ_(t,h,i)/∂C_hat_(t,h,i,k) are computed by autograd for all 26 concept coordinates, 32 evenly spaced held-out origins, five forecast leads, 32 assets and all five training seeds (25,600 derivatives per coordinate and fold). These are conditional responses of the fitted decoder, holding the evolved graph, other concept coordinates, observed market state, lead embedding and fitted parameters fixed. The plotted means include the original validation-fitted positive scalar. Values are decimal-return derivatives multiplied by 10,000, hence basis points per robust-standardized concept unit. Dots show medians, thick segments interquartile ranges, and thin segments 5th–95th percentiles of the sampled derivatives. All panels use the same signed horizontal scale.

## R4_direct_mean_magnitude

Magnitude of direct mean responses using the same derivative sample as R3. Bars show the mean absolute own-asset derivative, averaged equally over sampled origins, leads, assets and seeds; five open points show each seed's corresponding mean absolute value. Coordinates are sorted once by the fold-1 pooled magnitude, and that ordering is retained in folds 2 and 3. All panels share an axis beginning at zero and the same units (basis points per robust-standardized concept unit). The derivative includes each checkpoint's original validation-fitted positive mean scalar, so differences across folds include differences in mean shrinkage.
