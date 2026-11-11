# Proposed uniform finite-horizon rollout-neighborhood bounds

This supplements, rather than changes, the paper's registered-domain theorem. It concerns two trajectories of the same frozen fitted model with the same observed market state and lead schedule. It does not bound the error to unknown true dynamics, estimate their transition defects, or supply the optional Wasserstein decoder constant.

## Domain and metric

Fix a nominal rollout `(C_h,A_h)`, `h=0,...,5`. Graphs are symmetric with zero diagonal. Use Frobenius norms for both state blocks. For fixed positive radii `R_C,R_A`, define `E_h` as the intersection of the registered concept/graph domains with `||C-C_h||_F <= R_C`, `||A-A_h||_F <= R_A`. Each intersection is convex. Market state and lead embedding are fixed, so their network input columns do not enter state Lipschitz constants.

For every pair of states in `E_h`, the following bounds give a nonnegative matrix `M_h` such that the block difference between their next states is bounded by `M_h` times their current block difference. They are uniform over this entire neighborhood; no sampled Jacobian norm is used to form them.

## Laplacian bounds, including zero edges

Let `N=32`, `c_N = sqrt((N-1)/2)+sqrt((N-1)/N)`. For a symmetric zero-diagonal matrix, each row has squared norm at most half its total squared Frobenius norm, and its spectral norm is at most `sqrt((N-1)/N)` times its Frobenius norm (trace zero and Cauchy-Schwarz on the remaining eigenvalues). Consequently both the signed and unsigned Laplacians satisfy

`||L(A)-L(A')||_2 <= c_N ||A-A'||_F`.

For the unsigned case apply the same inequalities to `|A|-|A'|`, whose Frobenius norm is bounded by `||A-A'||_F`.

For fixed concept matrix `C`, define a sharper uniform bound `T(C)` on `A -> L_typed(A)C`. Represent the independent edge perturbations by `u_e=sqrt(2) dA_ij`, so `||u||_2=||dA||_F`. On a fixed edge-sign orthant, the signed-incidence formula gives, for directional coordinate `k`,

`J_A[L_s(A)C_k] u = B_s diag((B_s^T C_k) sign(a_e)) u/sqrt(2)`.

The sign diagonal has norm one and `||B_s||_2 <= sqrt(2(N-1))`. Let `p_k` be the sum of the two largest values of `|C_ik|` in that column. For unsigned coordinates let `w_k=max_i C_ik-min_i C_ik`; complete ordinary incidence has norm `sqrt(N)`. Thus one valid bound is

`T_inc(C)^2 = (N-1) sum_directional p_k^2 + (N/2) sum_unsigned w_k^2`.

A second valid bound follows by centering every unsigned column at its midrange. Write the centered combined matrix as `Ctilde=[C_dir, C_unsigned_centered]`. The degree action is bounded by `sqrt(N-1) max_i ||Ctilde_i||_2`; the adjacency actions are bounded by `sqrt(||C_dir||_2^2+||C_unsigned_centered||_2^2)`. Their sum is `T_alt(C)`. Set `T(C)=min(T_inc,T_alt)`.

Each orthant bound is uniform in edge magnitudes. The graph segment has finitely many sign changes, and the map is continuous at zero, so integrating along the segment establishes the same bound across sign changes. Edges identically zero along a segment contribute zero. No derivative choice at zero is needed. For `||C-C_h||_F<=R_C`, a uniform bound is `T(C_h)+c_N R_C` by splitting off the concept perturbation and applying the Laplacian-difference bound to each typed coordinate block.

## Residual and graph-proposal network bounds

Every MLP is Linear-SiLU-Linear-SiLU-Linear; the residual also ends with tanh. Use the uniform derivative bound `sup |SiLU'| < 1.1` and `sup |tanh'| <= 1`. Split the first residual weight into concept and neighborhood columns `W_C,W_N`; the market/lead columns are fixed. Let `K_F = 1.1^2 ||W_2||_2 ||W_3||_2`.

For symmetric `A`, the map `dC -> dC W_C^T + A dC W_N^T` has norm `max_{lambda in spectrum(A)} ||W_C+lambda W_N||_2`. At the center, let `a_min,a_max` be the extreme eigenvalues of `A_h`. On `E_h`, all eigenvalues lie in `[a_min-R_A,a_max+R_A]`. Convexity of the spectral norm gives the uniform first-layer concept bound `V_C`, the maximum of the two endpoint norms. The graph-input bound is `||W_N||_2 (||C_h||_2+R_C)`. Therefore

`L_FC=r_F K_F V_C`,
`L_FA=r_F K_F ||W_N||_2 (||C_h||_2+R_C)`.

For each graph encoder, use `L_U=1.1^2 ||W_3||_2 ||W_2||_2 ||W_1[:,concept]||_2` (and similarly `L_V`). Its output matrix on the neighborhood satisfies `||U(C)||_2 <= ||U(C_h)||_2 + L_U R_C`. The Gram-score map therefore has the uniform C-Lipschitz bound

`L_score = 2[(||U_h||_2+L_U R_C)L_U + (||V_h||_2+L_V R_C)L_V]/sqrt(d_g)`.

For the entrywise graph gate `g(z)=tanh(z) sigmoid((|z|-c)/tau)`, symmetry makes its derivative even away from zero. For `x>=0`, put `u=(x-c)/tau`. Since `tanh(x)<=x` and `sech^2(x)<=1`,

`g'(x) <= sigmoid(u)+u sigmoid'(u)+(c/tau) sigmoid'(u) <= 1.1+c/(4tau)`.

The function is continuously differentiable at zero; its derivative there is `sigmoid(-c/tau)`. Use the minimum of this bound and the paper's `1+1/(4tau)`. The configured `c=.10,tau=.05` gives the analytic upper bound `1.6` instead of `6`.

## Transition matrix and mean readout

At the fixed market state, damping coefficients `gamma_k` and scalar diffusion `D` are fixed. Compute the exact norm `ell_h` of the unprojected linear concept map at the center using the extreme eigenvalues of the appropriate signed/unsigned Laplacian for each coordinate. Then

`m_CC = ell_h + Delta D c_N R_A + Delta L_FC`,
`m_CA = Delta[D(T(C_h)+c_N R_C)+L_FA]`,
`m_AC = eta_A L_gate L_score`,
`m_AA = 1-eta_A`.

Coordinate clipping is nonexpansive. Symmetrization, diagonal removal, and radial projection onto the graph Frobenius ball compose to the Euclidean projection onto the convex registered graph domain and are nonexpansive. Thus these constants also bound the implemented projected update.

The calibrated mean readout has an analogous uniform bound. With the mean head's `W_C,W_N,K_mu`, let `V_mu,C` be the corresponding endpoint norm and `b=varrho max_i a_i^r M_mu K_mu`. Then

`||mu(C,A)-mu(C',A')||_2 <= b V_mu,C ||C-C'||_F + b ||W_N||_2 (||C_h||_2+R_C)||A-A'||_F`.

This is a deterministic forecast-mean bound, not the distribution-kernel `L_D` appearing in Eq.73. Calibration is held fixed and included explicitly.

## Finite-horizon certificate

Let `R=(R_C,R_A)`, `D_R=diag(R)`, and `P_h=M_{h-1}...M_0`. If `max_h ||D_R^{-1} P_h D_R||_infinity <= 1/alpha`, any admissible initial perturbation with block errors at most `alpha R` remains in all five registered neighborhoods: induction applies the uniform bound on each neighborhood, then checks `P_h alpha R <= R`. The certified amplification in the weighted block-max norm at each lead is `||D_R^{-1} P_h D_R||_infinity`. Alternatively, if `q=max_h ||D_R^{-1} M_h D_R||_infinity <1`, the same neighborhood radii are invariant from `alpha=1`, with the stronger bound `q^h`.

Primary diagnostic specification, before reading complete results: all 15 archived calibrated checkpoints; 32 evenly spaced held-out origins per fold; all five leads; fixed `R_C=.1,R_A=.05`; fixed initial fraction `alpha=.1`. Secondary radius grid `.001,.01,.1,1` with the same graph/concept radius ratio `.5` will be reported completely. No labels, fitting, tuning, or test-driven checkpoint choice. Original registered-domain constants remain reported separately.

Evaluation uses float64 spectral calculations and a declared relative `1e-10` plus absolute `1e-12` numerical cushion. This is an analytic bound evaluated numerically, not an interval-arithmetic or formal machine-verified proof. Independent dense-Jacobian and finite-perturbation checks validate implementation without defining or replacing the bounds.
