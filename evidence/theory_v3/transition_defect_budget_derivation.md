# Finite-horizon containment with transition residuals

Let the frozen nominal trajectory satisfy $z_{h+1}=F_h(z_h)$, and let $\bar z_h$ be an **admissible comparison trajectory**: each of its concept and graph blocks lies in the registered domain. Market state, parameters, and lead schedule are fixed. Define its one-step residual by $r_h=\bar z_{h+1}-F_h(\bar z_h)$. This analysis assumes a bound on that residual; it does not estimate residuals to the unknown true dynamics or assert that observed future-state errors satisfy it.

For the previously certified product neighborhood at lead $h$, let $M_h$ be its nonnegative uniform two-block Lipschitz bound. Write $R=(R_C,R_A)^\top$ and let $\delta_h$ contain the two Frobenius errors. Whenever $\bar z_h$ belongs to the neighborhood,

$$
\delta_{h+1}\le M_h\delta_h+\epsilon_h,
\qquad
\epsilon_h=(\|r_h^C\|_F,\|r_h^A\|_F)^\top.
$$

For $\delta_0\le\alpha R$ and $\epsilon_h\le\beta R$, define

$$
P_0=I,\quad S_0=0,\qquad
P_{h+1}=M_hP_h,\quad S_{h+1}=M_hS_h+I.
$$

The candidate bound is $v_h=\alpha P_hR+\beta S_hR$. If $v_h\le R$ at every prefix, induction proves both $\delta_h\le v_h$ and membership in each neighborhood. The radius condition and the assumed registered-domain admissibility together establish membership in the intersection used by the uniform local bound.

The largest uniform residual fraction allowed by these particular stored matrices is

$$
\beta_{\max}
=\min_{1\le h\le5,\,i\in\{C,A\}}
\frac{[R-\alpha P_hR]_i}{[S_hR]_i}.
$$

The minimum is taken over all prefixes, not only the final lead. A downward numerical cushion is applied before reporting a common budget over trajectories. The computed budgets are sufficient conditions derived from the analytic upper bounds; they are not necessary stability thresholds of the actual model.

The completed evaluation uses the same 15 checkpoints, 32 origins per fold, four radii and matrices as the state audit. With $\alpha=0.1$ and $\beta=0.05$, all 1,920 trajectory/radius cases satisfy containment. For the primary radii $(0.1,0.05)$, the maximum normalized error bound over all five prefixes and all trajectories is at most $0.750829$, $0.842103$, and $0.447400$ in folds 1, 2, and 3, respectively (rounded upward). The common admissible residual fractions, rounded downward to six decimal places, are $0.078698$, $0.066577$, and $0.143265$.

An alternative that automatically preserves admissibility is a projected perturbed update $\bar z_{h+1}=\Pi(F_h(\bar z_h)+u_h)$ with the same block projections. Since $F_h(\bar z_h)$ is already admissible, nonexpansiveness gives $\|r_h^C\|_F\le\|u_h^C\|_F$ and $\|r_h^A\|_F\le\|u_h^A\|_F$. Thus the same result applies when each projected-update forcing block is bounded by $\beta R$.

Code: `audit_transition_defect_budgets.py`. Full numerical records: `transition_defect_results/transition_defect_budgets.json`. Independent verification explicitly expands the time-ordered tail products, rather than reusing the matrix-sum recurrence.
