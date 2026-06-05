"""
Recreates Table 7.2 from the thesis:
    Quantile-expectile correspondence for Z ~ N(0, 1)

Two methods are implemented:
  1. Analytic  — solve the balance equation (7.8) with scipy.optimize.brentq
  2. Sample    — fixed-point weighted-mean iteration on 5e6 N(0,1) draws (another method not in thesis)

Balance equation (7.8):
    (2*tau - 1) * phi(m) = m * [tau + (1 - 2*tau) * Phi(m)]

Equivalent to the first-order condition of the asymmetric squared loss (7.3):
    tau * E[(Z - m)+] = (1 - tau) * E[(m - Z)+]

where, for Z ~ N(0, 1):
    E[(Z - m)+] = phi(m) - m * (1 - Phi(m))        (7.6)
    E[(m - Z)+] = m * Phi(m) + phi(m)               (7.7)
"""

import numpy as np
from scipy import stats, optimize


# ---------------------------------------------------------------------------
# Method 1: analytic — solve balance equation via Brent's method
# ---------------------------------------------------------------------------

def _balance_equation(m: float, tau: float) -> float:
    """
    Returns  tau*E[(Z-m)+] - (1-tau)*E[(m-Z)+]  for Z ~ N(0,1).
    This is zero at m = mu_tau(Z).
    Uses closed-form expressions (7.6)-(7.7) from the thesis.
    """
    phi = stats.norm.pdf(m)           # phi(m)
    Phi = stats.norm.cdf(m)           # Phi(m)

    E_up   = phi - m * (1.0 - Phi)   # E[(Z - m)+]   eq. (7.6)
    E_down = m * Phi + phi            # E[(m - Z)+]   eq. (7.7)

    return tau * E_up - (1.0 - tau) * E_down


def expectile_analytic(tau: float, tol: float = 1e-12) -> float:
    """
    Compute the tau-expectile of N(0,1) by solving eq. (7.8) with
    Brent's method.  Returns 0.0 exactly for tau = 0.5.
    """
    if tau == 0.5:
        return 0.0
    return optimize.brentq(_balance_equation, -10.0, 10.0,
                            args=(tau,), xtol=tol)


# ---------------------------------------------------------------------------
# Method 2: sample-based; fixed-point iteration on N(0,1) draws
# ---------------------------------------------------------------------------

def expectile_sample(tau: float, z: np.ndarray,
                     tol: float = 1e-10, max_iter: int = 2000) -> float:
    """
    Compute the tau-expectile via another method.
    It is called weighted-mean fixed-point iteration:

        m_{k+1} = E[w_k * Z] / E[w_k]
        w_k     = tau * 1{Z > m_k} + (1 - tau) * 1{Z <= m_k}

    """
    m = 0.0
    for _ in range(max_iter):
        # asymmetric weights: heavier on the side containing fewer observations
        w = np.where(z > m, tau, 1.0 - tau)
        m_new = (w * z).sum() / w.sum()
        if abs(m_new - m) < tol:
            return m_new
        m = m_new
    # if we exit without converging, return best estimate
    return m


# ---------------------------------------------------------------------------
# Build Table 7.2
# ---------------------------------------------------------------------------

# Expectile levels
TAU_LEVELS = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50]

# Draw 5 × 10^6 standard normal samples
rng = np.random.default_rng(seed=42)
Z = rng.standard_normal(5_000_000)

# Collect results
rows = []
for tau in TAU_LEVELS:
    mu_analytic = expectile_analytic(tau)
    mu_sample   = expectile_sample(tau, Z)
    alpha       = float(stats.norm.cdf(mu_analytic))   # alpha = Phi(mu_tau)
    rows.append({
        "tau":          tau,
        "mu_analytic":  mu_analytic,
        "mu_sample":    mu_sample,
        "alpha":        alpha,
    })

# ---------------------------------------------------------------------------
# Print the table
# ---------------------------------------------------------------------------

header = (
    f"{'tau':>8}  "
    f"{'mu_tau (analytic)':>19}  "
    f"{'mu_tau (sample)':>17}  "
    f"{'alpha = Phi(mu_tau)':>21}"
)
sep = "-" * len(header)

print()
print("Table 7.2 — Quantile-expectile correspondence for Z ~ N(0, 1)")
print(sep)
print(header)
print(sep)
for r in rows:
    print(
        f"{r['tau']:>8.3f}  "
        f"{r['mu_analytic']:>19.4f}  "
        f"{r['mu_sample']:>17.4f}  "
        f"{r['alpha']:>21.4f}"
    )
print(sep)
print()
print("Notes:")
print("  mu_tau (analytic) : root of the balance equation solved via Brent's method")
print("  mu_tau (sample)   : fixed-point weighted-mean iteration, n = 5,000,000 draws")
print("  alpha = Phi(mu_tau): the quantile level q_alpha(Z) = mu_tau(Z)")
print()
