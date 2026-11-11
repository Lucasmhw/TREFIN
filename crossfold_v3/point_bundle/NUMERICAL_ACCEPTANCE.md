# Numerical evaluation decision before full optimized training/testing

**Decision:** proceed with the ordered, grouped independent-origin evaluator using a fixed outer inference batch of 32. Training remains batch 64 and uses the untouched original forward method. This decision concerns numerical implementation agreement, not predictive performance, and was made before any optimized full run or test evaluation.

The complete CUDA diagnostic is preserved in `tests/timemixerpp_cuda_diagnostic.json`, including its explicit `all_original_thresholds_pass=false`. The earlier absolute normalized threshold of 2e−5 was exceeded; it has not been relabeled as passed. The replacement practical criterion, numerical justification and immutable hashes are recorded in `numerical_acceptance.json`, which is checked against the source manifest and copied into every protocol.

The registered criterion requires all of:

- Maximum error in actual return units ≤1e−6, equivalent to 0.01 basis point.
- Maximum error in normalized outputs ≤1e−4.
- RMS output error divided by reference RMS ≤1e−4 (0.01%).
- Finite outputs, exact tested training losses/gradients, and identical tested FFT selection membership and order.

The diagnostic used 32 evenly spaced validation histories and three parameter states: initialization, two actual training updates, and the existing original runner's fitted seed-1 checkpoint. No test target or performance ranking was used. Every batch-1 comparison is bitwise exact. With batches 4/16/32, all FFT input values, frequency scores, selection memberships and selection orders are exact at both mixer blocks. Differences first appear in finer-resolution block outputs and then propagate to the coarse final readout. This pattern is consistent with batch-shape-dependent CUDA float32 convolution arithmetic, rather than another origin's information entering the FFT selection. Both matmul and cuDNN TF32 flags were already disabled.

Across all checked states and batches, maximum normalized output difference is 3.70741e−5, maximum raw-return difference is 6.16596e−7 (0.00617 basis point), and maximum RMS-relative difference is 2.95327e−5 (0.00295%). All pass the stated observable-unit and relative-scale criteria. Training losses and every parameter gradient are exactly identical in the tested states. CPU comparisons are bitwise identical across all tested batch sizes.

The first evaluator version merged weights for duplicate periods. That mathematically valid reassociation produced additional roundoff before layer normalization. It was replaced before deployment by restoring the original top-k rank order and calling the unchanged native stack/softmax/multiply/sum procedure. The accepted module hash is `e14c45197d41b0f9115c2498de0622773b82e29aeae4eef82c5d39315558923b`.

This is an explicitly documented practical agreement criterion, not a claim of bitwise CUDA equivalence or a bound for every possible input. The model architecture, parameters, training objective, training execution, candidate budget, data splits, residual scenario construction and calibration family remain unchanged. Evaluation batch 32 is fixed because it was tested; untested inference batch 64 is not used. The fresh experiment's own chosen checkpoint must still reload and reproduce its validation ES under this fixed evaluator, as enforced by the runner.

The CUDA timings showed approximately 3.83× faster inference for the real fitted checkpoint at batch 32 in this diagnostic. That is a measured local timing, not a promised whole-training speedup. The original unaccelerated first epoch/checkpoint and all numerical diagnostic failures remain separate evidence; no old experiment or main-table result is overwritten.
