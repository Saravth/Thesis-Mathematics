"""
Landscape of $s_{\alpha}(v, e)$ based on the Tail-GAN score function

The Figures created are like Figure 1 from the article
'Tail-GAN: Learning to Simulate Tail Risk Scenarios (Cont et al., 2025).

"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.05
# np.random.seed(123)  # reproducibility
x_samples = np.random.uniform(-1, 1, size=1000)  # Uniform distribution
W_alpha = 10

# Grids for v and e
v_values = np.linspace(-2.5, 0, 200)
e_values = np.linspace(-2.5, 0, 200)
V, E = np.meshgrid(v_values, e_values)

X = x_samples[:, None, None]
Vb = V[None, :, :]
Eb = E[None, :, :]

indicator = (X <= Vb).astype(float)
term1 = (W_alpha / 2) * (indicator - alpha) * (X**2 - Vb**2)
term2 = indicator * Eb * (Vb - X)
term3 = alpha * Eb * ((Eb / 2) - Vb)
S = (term1 + term2 + term3).mean(axis=0)
S = np.clip(S, 0, None)

# -----------------------------------------------------------------------------
#  Plots
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])

# (a) 3D surface plot
ax1 = fig.add_subplot(gs[0], projection='3d')
surf = ax1.plot_surface(V, E, S, cmap='viridis', edgecolor='none', alpha=0.8)
ax1.set_title(r"(a) $s_\alpha(v,e)$")
ax1.set_xlabel("$v$ (VaR)")
ax1.set_ylabel("$e$ (ES)")
ax1.set_zlabel(r"$s_{\alpha}(v,e)$")
ax1.set_zlim([0, 0.8])



# (b) S(v,e) vs e for fixed v
ax2 = fig.add_subplot(gs[1])
v_fixed = [-1.2, -1.0, -0.8, -0.5]
for v in v_fixed:
    # find nearest index of v
    idx_v = np.argmin(np.abs(v_values - v))
    s_e = S[:, idx_v]  # all e for this v
    line, = ax2.plot(e_values, s_e, label=fr"$v={v}$")
    min_index = np.argmin(s_e)
    ax2.scatter(e_values[min_index], s_e[min_index],
                color=line.get_color(), marker='*', s=80)
ax2.set_title(r"(b) $s_\alpha(v,e)$ as a function of $e$ with given $v$")
ax2.set_xlabel("e")
ax2.set_ylabel(r"$s_\alpha(v,e)$")
ax2.legend()

# (c) S(v,e) vs v for fixed e
ax3 = fig.add_subplot(gs[2])
e_fixed = [-1.2, -1.0, -0.8, -0.5]
for e in e_fixed:
    # find nearest index of e
    idx_e = np.argmin(np.abs(e_values - e))
    s_v = S[idx_e, :]  # all v for this e
    line, = ax3.plot(v_values, s_v, label=fr"$e={e}$")
    min_index = np.argmin(s_v)
    ax3.scatter(v_values[min_index], s_v[min_index],
                color=line.get_color(), marker='*', s=80)
ax3.set_title(r"(c) $s_\alpha(v,e)$ as a function of $v$ with given $e$")
ax3.set_xlabel("v")
ax3.set_ylabel(r"$s_\alpha(v,e)$")
ax3.legend()

plt.tight_layout()
plt.show()
