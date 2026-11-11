# Explicit uniform local decoder-kernel W1 constants

This is a one-lead statement on the 32-asset return vector, with Euclidean transport cost. Fix the market state, lead, fitted parameters, and positive mean scalar. The state domain is the same product neighborhood intersected with registered domains as in `tube_derivation.md`. The result compares two laws produced by the same decoder. A bound on the decoder's discrepancy from the unknown true conditional law remains a separate quantity.

## 1. Uniform network and parameter bounds

The spectral first-layer endpoint bounds from `tube_bounds.py` give uniform two-block Lipschitz constants for each scalar-output MLP. Free-feature heads mask risk coordinates in both first-layer concept and neighborhood blocks. Pooled lambda/nu heads include the row-averaging factor. Mean constants alone include the fitted positive scalar; scale, precision, and nu constants do not.

The additional `local_mlp_bounds.py` tightens these bounds using a center Jacobian **plus a uniform derivative-variation bound**, rather than treating a point Jacobian as a certificate. For each first-layer preactivation, write its bilinear state dependence as `v^T C w_C + v^T A C w_N`, with `v=e_i` for a node head and `v=1/N` for a pooled head. The first-order C sensitivity at the center is the Frobenius norm of `v w_C^T + A_h v w_N^T`. The first-order graph sensitivity is the Frobenius norm of the symmetric, zero-diagonal part of `v (C_h w_N)^T`. The bilinear remainder is at most `R_C R_A ||v|| ||w_N||`; for a node head, symmetry sharpens `||v^T Delta A||` to `R_A/sqrt(2)`. These yield a preactivation interval at every hidden unit.

SiLU has derivative in `[-.1,1.1]` and second derivative bounded in magnitude by `.75`. One proof of the first upper bound reduces it to the convex inequality `1.1 exp(-x)+.1 exp(x)+1.2-x >=0`, whose minimum is `2.4-log(11)>0`; the lower bound follows from `f'(-x)=1-f'(x)`. For the second bound, `2 sigmoid' <=.5` and `|x| sigmoid' <=.25` follow from `cosh x >=1+x^2/2`. Hidden-activation intervals propagate through the second affine layer. The SiLU value interval also uses its unique minimum in `[-2,-1]` and the rigorous lower bound `-1/e`.

For the two hidden derivative diagonals, expand

`J-J_0 = W_3 Delta D_2 W_2 D_{1,0} W_1 + W_3 D_{2,0} W_2 Delta D_1 W_1 + W_3 Delta D_2 W_2 Delta D_1 W_1`.

Absolute coefficient bounds give a componentwise error box for the input gradient. The exact center state-Jacobian block norms plus this gradient error and the corresponding C/A variation terms bound every state Jacobian throughout the neighborhood. Integrating on the product domain gives separate C- and A-Lipschitz bounds. The minimum with the original spectral bound is valid for each block. Direct output intervals from the same hidden intervals additionally tighten the softplus argument ranges.

For a pre-softplus argument interval `[f_i^-,f_i^+]`, monotonicity gives output intervals and the slope bound `sigmoid(max_i f_i^+)`. The downside-scale argument includes its exact linear risk signal. These steps give parameter bounds for mean, both scales, q, lambda, and nu. Let their block constants be `ell_mu`, `ell_+`, `ell_-`, `ell_q`, `ell_lambda`, `ell_nu`, and let `nu_lo>2`, `sigma_max`, and `D_sigma` bound the minimum nu, maximum asymmetric scale, and Euclidean norm of `sigma_+-sigma_-` throughout the neighborhood.

## 2. Precision and covariance-factor coupling

The implemented precision is `Q=diag(q)+lambda L_s(A)+epsilon_Q I`, with `q_i>=epsilon_Q` and positive lambda. Define a retained signed graph by preserving the center sign and lowering each magnitude to `max(|A_h,ij|-R_A/sqrt(2),0)`. Each retained edge keeps that sign and at least that magnitude for every admissible graph in the neighborhood. Every omitted signed-edge contribution is positive semidefinite. Consequently

`Q >= diag(q_lo)+lambda_lo L_s(A_retained)+epsilon_Q I`.

Its smallest eigenvalue gives a uniform positive floor `m_Q`. A uniform diagonal upper bound is obtained from

`Q_ii <= q_hi,i + epsilon_Q + lambda_hi [degree_i(A_h)+sqrt((N-1)/2) R_A]`.

Write its maximum as `qdiag_hi`. Since `(Q^-1)_ii>=1/Q_ii`, every covariance diagonal is at least `1/qdiag_hi`. Uniform precision-block constants follow from

`||Delta Q||_F <= ||Delta q||_2 + Ls_F_hi |Delta lambda| + lambda_hi sqrt(N)||Delta A||_F`,

where `Ls_F_hi=||L_s(A_h)||_F+sqrt(N)R_A`. The factor `sqrt(N)` follows by squaring the Frobenius contributions of the diagonal degree and off-diagonal adjacency differences.

Let `V=Q^-1/2` be the principal inverse square root. Define `F(Q)` by normalizing row i of V by `max(||V_i||,sqrt(diagonal_floor))`. Then `F F^T` is exactly the covariance-normalized correlation used by the code before jitter. On the convex precision segment, the inverse-square-root Frechet bound is `||dV||_F <= ||dQ||_F/(2 m_Q^(3/2))`. The row normalization/clipped-normalization derivative has norm at most `1/sqrt(d_min)`, where `d_min=max(1/qdiag_hi,diagonal_floor)`. Therefore

`||Delta F||_F <= ||Delta Q||_F/[2 sqrt(d_min) m_Q^(3/2)]`.

The same lower/diagonal bounds persist on the precision segment. This is a factor coupling: F need not equal the Cholesky factor, since it produces the same Gaussian covariance.

When the original covariance-diagonal clamp is inactive throughout the neighborhood, correlation is also invariant to the positive scalar normalization `Q -> Q/lambda`. The equivalent precision is `L_s(A)+diag((q+epsilon_Q)/lambda)`. Bound its diagonal ratio and retain the same stable-edge graph; the lambda contribution then enters only that diagonal ratio. This gives a second uniform factor-bound vector. Taking the smaller bound separately for C and A is justified by telescoping through `(C',A)`, which remains in the product domain. Both analytic routes are retained in the output.

## 3. Gamma-mixture degrees-of-freedom bound, including the sampler floor

Let `chi~Gamma(nu/2,rate=1/2)`, `G~N(0,B)` independently, and

`xi_nu=sqrt(nu-2) G/sqrt(max(chi,t))`, for a fixed floor `t>=0`.

The chi-density nu-score is `s=.5(log chi-log2-psi(nu/2))`, with `E s^2=.25 trigamma(nu/2)`. Also `E[1/max(chi,t)]<=1/(nu-2)`. For a 1-Lipschitz test function anchored at zero, differentiating expectation gives an explicit scaling term and a density-score term. The scaling term is at most `sqrt(trace B)/[2(nu-2)]`. Independence and Cauchy-Schwarz bound the score term by `.5 sqrt(trace B * trigamma(nu/2))`. Thus, uniformly for `nu>=nu_lo>2`,

`L_nu(B) <= sqrt(trace B) [1/(2(nu_lo-2))+.5 sqrt(trigamma(nu_lo/2))]`.

On each compact nu interval, Gamma/Gaussian domination justifies differentiation: near zero the envelope is proportional to `chi^(nu_lo/2-3/2)(1+|log chi|)`, which is integrable. For general Lipschitz test functions, smooth anchored mollifications and dominated convergence give the same inequality. The Kantorovich dual then gives the W1 bound. Trigamma decreases, so the lower endpoint controls the interval. This argument includes t=0 and the implemented `t=1e-8`.

The mathematical law has B=R and no jitter. The implemented sampler has `B=R+2e-5 I`, since decoding and sampling each add `1e-5 I`. In either case `trace B<=N(1+j)` with `j=0` or `2e-5`; this remains true when the covariance diagonal clamp activates. The Gaussian dimension is N=32 at one lead; the code shares one chi draw across those assets. A joint five-lead law would require a separate product-law coupling.

The deterministic centering coefficient in the code is the ideal univariate Student coefficient `kappa(nu)=.5 E|xi_nu,ideal|`. Consequently `kappa<=.5` and its Lipschitz constant is at most half the same d=1, floor-zero Gamma bound. This bounds the function actually subtracted by the implementation without identifying it with the jittered/floored sampler's positive-part mean.

## 4. Complete local decoder constant

Write `h_sigma(x)=sigma_+ x_+ - sigma_-(-x)_+` coordinatewise. Its x-Lipschitz constant is at most `sigma_max`. For fixed nu, couple two precisions by `F(Q)Z+sqrt(j)Z'` with the same independent Gaussian pair and the same chi variable. The jitter term cancels. Since the standardized, floored scale mixture has covariance at most I, the precision contribution to W1 is at most `sigma_max ||Delta F||_F`.

For scale changes at fixed latent law, symmetry and disjoint positive/negative parts give the bound

`sqrt((1+j)/2) sqrt(||Delta sigma_+||_2^2+||Delta sigma_-||_2^2)`.

The deterministic centering adds at most `.5(||Delta sigma_+||_2+||Delta sigma_-||_2)`. For nu changes, the Gamma bound contributes `sigma_max L_nu(B)|Delta nu|` and the centering function contributes `D_sigma L_kappa |Delta nu|`. The mean contributes `||Delta mu||_2`.

Adding these terms gives an explicit block vector `ell_D=(L_DC,L_DA)` satisfying

`W1(D(C,A),D(C',A')) <= L_DC ||Delta C||_F + L_DA ||Delta A||_F`

uniformly on the specified neighborhood. A constant under the paper's sum-block norm is `L_D=max(L_DC,L_DA)`. For a certified rollout tube, compose the actual lead-specific vector with the state product:

`W1(D_h(z_h),D_h(z'_h)) <= ell_D,h^T P_h delta_0`.

This is an explicit local distribution-sensitivity bound. It leaves the global registered-domain extension and the reference-law defect in Eq.73 as their separately stated objects. The complete audit retains every radius, both laws, and both the spectral and center-plus-variation network bounds, including conservative values. No parameter tuning or true-law error estimation is performed.

For an admissible comparison path with additive transition defects, the independently checked extension in `transition_defect_budget_derivation.md` supplies a certified state-error vector `v_h=.1 P_h R+beta S_h R`. The same decoder proof then gives `W1 <= ell_D,h^T v_h`. The saved decoder NPZ's propagated field uses `v_h=P_h(.1R)` with zero additive defects; the nonzero-defect expression is a separate composition and is not the value stored in that field. Comparison with a true conditional return law additionally includes its reference decoder defect.
