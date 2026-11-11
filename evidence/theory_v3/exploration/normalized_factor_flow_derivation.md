# Exploratory normalized Gaussian factor-flow bound

This is an approved analytic supplement, independently reviewed and retained alongside the earlier valid inverse-root certificate. It concerns the same product neighborhoods, fixed market/lead, and one-lead return law. No smaller neighborhood or fitted Jacobian is substituted.

## 1. A covariance-factor flow with normalization cancellation

Let Q(t) be a differentiable SPD path, Sigma=Q^-1, H=Q', and choose a factor V with VV^T=Sigma that solves V'=-Sigma H V/2. Differentiating VV^T shows it follows Sigma'=-Sigma H Sigma. Row normalization F_i=V_i/sqrt(Sigma_ii) gives correlation FF^T. Its exact row speed is

||F_i'||^2 = (1/4)[u_i^T T^2 u_i-(u_i^T T u_i)^2],

where T=Sigma^(1/2) H Sigma^(1/2) and u_i=Sigma^(1/2)e_i/sqrt(Sigma_ii) is a unit vector. This is invariant to adding a scalar multiple of I to T. Since sum_i u_i u_i^T has the same spectrum as the correlation R,

||F'||_F <= (sqrt(lambda_max(R))/2)||T||_F.

The variance of a symmetric matrix with spectrum in [a,b] is at most (b-a)^2/4, hence also

||F'||_F <= sqrt(N)(b-a)/4.

The path coupling F(t)Z, with shared Z~N(0,I), bounds Gaussian W2 by the integral of this factor speed. A fixed additive jitter can be coupled by shared independent noise and cancels. A shared fixed-nu scale mixture with the implemented chi floor has second scale moment at most one, so the same Gaussian factor bound controls that mixture's coupling. The asymmetric return map then multiplies it by sigma_max. This does not require the chosen endpoint factor to be a fixed principal-root function.

## 2. Remove positive precision scale, then separate parameter paths

When the normalization clamp is inactive, write equivalent precision T0=L_s(A)+diag(b), b_i=(q_i+epsilon)/lambda. Throughout the product parameter box use K=L_s(A_retained)+diag(b_lower)>0 as a common Loewner lower bound. The telescoping argument separates q, lambda and A through an enlarged parameter box. Its intermediate combinations need not be jointly attainable from a state: the q and lambda intervals and the graph neighborhood independently imply the same K and precision-diagonal upper bounds throughout that box. The original precision diagonal upper bound also keeps its covariance-normalization clamp inactive on all intermediate combinations.

For q-only changes, v_i=(log(q_i+epsilon))'. The relative precision derivative is W diag(v) W^T, W=T0^-1/2 diag(sqrt(b)), with ||W||_2<=1. Consequently its Frobenius norm is <=||v||_2. Its spectral range is at most max(v,0)-min(v,0)<=sqrt(2)||v||_2. The factor-speed coefficient is therefore min(sqrt(r_upper)/2,sqrt(N/8))||v||_2.

For lambda-only changes, the relative derivative is -(log lambda)' W W^T, with spectrum contained in the interval between 0 and -(log lambda)'. The factor-speed bound is sqrt(N)/4 times |(log lambda)'|. This uses the row-normalization cancellation and avoids multiplying this scalar change by sqrt(N) twice.

Both log-head Lipschitz constants are bounded by their raw argument-network constants because d log(c+softplus(x))/dx <=1 for c>=0. Indeed, for t=exp(x), log(1+t)>=t/(1+t), so softplus(x)>=sigmoid(x). The minimum with softplus-head constants divided by the positive parameter lower bound gives another valid option.

## 3. A uniform weighted graph-derivative bound

Use edge coordinates u_e=sqrt(2) A_ij, so their Euclidean norm equals the symmetric zero-diagonal graph Frobenius norm. On a fixed edge-sign region,

L_s'(A)=sum_e sign(A_e) u_e' b_e b_e^T/sqrt(2),

with b_e=e_i-sign(A_e)e_j. Since T0>=K, congruence contraction gives

||T0^-1/2 L_s' T0^-1/2||_F <= ||K^-1/2 L_s' K^-1/2||_F.

For a specified edge-sign vector, the squared operator norm of the latter edge-to-matrix linear map is the largest eigenvalue of G_ef=(b_e^T K^-1 b_f)^2/2 (diagonal sign factors do not change its eigenvalues). Determine each edge's allowed sign set from the full neighborhood: sign is fixed if |A_center,ij|>R_A/sqrt(2), and both signs are allowed otherwise. Define the nonnegative symmetric envelope U_ef by maximizing G_ef over the allowed sign pair. Every possible G is entrywise <=U, so its spectral radius is <=lambda_max(U). Thus

||relative graph derivative||_F <=sqrt(lambda_max(U))||Delta A||_F.

Combining the Frobenius and spectral-range factor-speed bounds gives graph coefficient min(sqrt(r_upper)/2,sqrt(N/8))*sqrt(lambda_max(U)). The simpler sqrt(N)/lambda_min(K) graph-relative bound is retained as a minimum. All edge-sign regions are covered uniformly; continuity/integration handles sign changes and zero edges.

One uniform correlation spectral bound is r_upper=min(N,lambda_max(D_upper^(1/2) K^-1 D_upper^(1/2))), where D_upper bounds each precision diagonal: inverse diagonal >=1/precision_diagonal. To justify the diagonal scaling, use the equivalent nonzero spectrum of K^-1/2 D K^-1/2; the diagonal inequality D<=D_upper then gives Loewner ordering. Sharper covariance diagonal lower bounds can be substituted if independently established.

## 4. Scope and completed verification

The resulting graph/q/lambda parameter coefficients compose with the already validated uniform head bounds and preserve the same primary radii R_C=.1,R_A=.05. Each valid alternative is a separate C/A bound on the product domain, so componentwise minima can be justified by telescoping in C and A. The complete cohort and all four radii are retained. Actual-checkpoint validation compares Gaussian W2, and the separate mathematical fixtures check the exact infinitesimal factor-flow identity; neither incorrectly demands the endpoint difference of the old principal-root factor, which is a different coupling.

Independent review passed all 9,600 readout neighborhoods, cellwise minima and state products, aggregates and hashes. Actual-checkpoint validation passed 960 dense log-head block-bound comparisons, 8,640 finite log-head comparisons, and 4,320 Gaussian W2 comparisons. Independent small-matrix fixtures passed 192 signed-region cases and 576 factor-flow identity checks. The analytic inequalities establish uniformity; the float64 calculations and these checks establish the reported numerical implementation scope. This supplement does not estimate a true-law prediction defect.
