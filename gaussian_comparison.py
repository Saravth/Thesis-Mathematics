"""
Generates all comparison figures for the synthetic Gaussian study.
Creates a folder  gaussian_comparison/  containing:

  gaussian_WGAN.png              histogram + Q-Q: WGAN-GP vs real
  gaussian_TAILGANalpha.png      histogram + Q-Q: Tail-GAN (alpha=0.05) vs real
  gaussian_TAILGANalphas.png     histogram + Q-Q: Tail-GAN (alpha=[0.01,0.05]) vs real
  gaussian_EXPECTILEGANtau.png  histogram + Q-Q: Expectile-GAN (tau=0.10) vs real
  gaussian_EXPECTILEGANtaus.png histogram + Q-Q: Expectile-GAN (tau=[0.01,0.05,0.10]) vs real
  tail_comparison.png            worst 15% left-tail across all models
  training_curves.png            2×3 loss and risk-estimate tracking
  sample_paths.png               5 random return paths per model (4 stacked panels)
  sample_losses_paths.png        5 worst-case return paths per model (4 stacked panels)

"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# imports from GAN modules
from wgan_gp import train_wgan_gp
from tail_gan import (
    Generator      as _TailGen,
    TailDiscriminator,
    FisslerZiegelScorer,
    calculate_var,
    calculate_es,
    get_device,
    train_tailgan_improved,
)
from expectile_gan import train_expectile_gan, compute_expectile

# global style
plt.rcParams.update({
    'font.size':        10,
    'axes.titlesize':   12,
    'axes.labelsize':   10,
    'legend.fontsize':   8,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.grid':        True,
    'grid.alpha':       0.30,
    'grid.linestyle':   '--',
})

# constants
OUTPUT_DIR = 'gaussian_comparison'

# colour palette shared across all figures
C = {
    'real':      '#1f77b4',   # blue
    'wgan':      '#ff7f0e',   # orange
    'tailgan':   '#2ca02c',   # green
    'expectile': '#d62728',   # red
}

# -----------------------------------------------------------------------------
#  Data
# -----------------------------------------------------------------------------
def make_gaussian(n=1000, T=50, std=0.02, seed=42):
    np.random.seed(seed)
    return np.random.normal(0.0, std, (n, T)).astype(np.float32)


# -----------------------------------------------------------------------------
#  Tail-GAN training with per-alpha history tracking.
# -----------------------------------------------------------------------------
def train_tailgan_tracked(data, n_epochs=500, batch_size=64, latent_dim=100,
                          lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
                          alphas=(0.05,), verbose=True):
    return train_tailgan_improved(
        data=data,
        n_epochs=n_epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        lr_d=lr_d,
        lr_g=lr_g,
        lambda_dual=lambda_dual,
        alphas=tuple(alphas),
        n_critic=5,
        verbose=verbose,
    )


def _train_tailgan_legacy(data, n_epochs=500, batch_size=64, latent_dim=100,
                          lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
                          alphas=(0.05,), verbose=True):
    """
    Train Tail-GAN and return a history dict in which
      history['var_estimate']  and  history['es_estimate']
    are  {alpha: [epoch_means]}  dicts.

    """
    device     = get_device()
    alphas     = list(alphas)
    output_dim = data.shape[1]

    data_mean  = float(data.mean())
    data_std   = float(data.std())
    data_norm  = (data - data_mean) / data_std

    generator     = _TailGen(latent_dim=latent_dim,
                              output_dim=output_dim).to(device)
    discriminator = TailDiscriminator(input_dim=output_dim,
                                      alphas=alphas).to(device)
    scorer        = FisslerZiegelScorer(alphas=alphas)

    opt_G = torch.optim.Adam(generator.parameters(),
                              lr=lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(),
                              lr=lr_d, betas=(0.5, 0.999))

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(data_norm)),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )

    history = {
        'd_loss':       [],
        'g_loss':       [],
        'var_estimate': {a: [] for a in alphas},
        'es_estimate':  {a: [] for a in alphas},
    }

    for epoch in range(n_epochs):
        d_buf, g_buf = [], []
        var_buf = {a: [] for a in alphas}
        es_buf  = {a: [] for a in alphas}

        for (real,) in loader:
            real = real.to(device)
            bs   = real.size(0)

            # discriminator
            opt_D.zero_grad()
            z        = torch.randn(bs, latent_dim, device=device)
            fake     = generator(z)
            fake_est = discriminator(fake.detach())
            real_est = discriminator(real)
            s_fake   = scorer(fake_est, real)
            s_real   = scorer(real_est, real)
            d_loss   = -(s_fake - lambda_dual * s_real)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            opt_D.step()
            d_buf.append(d_loss.item())

            # generator
            opt_G.zero_grad()
            z2    = torch.randn(bs, latent_dim, device=device)
            fake2 = generator(z2)
            g_loss = scorer(discriminator(fake2), real)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_G.step()
            g_buf.append(g_loss.item())

            # per-alpha estimate tracking
            with torch.no_grad():
                for i, a in enumerate(alphas):
                    # fake_est columns looks like: [VaR_a0, ES_a0, VaR_a1, ES_a1, ...]
                    v = (fake_est[:, 2 * i].mean().item()
                         * data_std + data_mean)
                    e = (fake_est[:, 2 * i + 1].mean().item()
                         * data_std + data_mean)
                    var_buf[a].append(v)
                    es_buf[a].append(e)

        history['d_loss'].append(float(np.mean(d_buf)))
        history['g_loss'].append(float(np.mean(g_buf)))
        for a in alphas:
            history['var_estimate'][a].append(float(np.mean(var_buf[a])))
            history['es_estimate'][a].append(float(np.mean(es_buf[a])))

        if verbose and (epoch + 1) % 50 == 0:
            parts = ' '.join(
                f"VaR({int(a*100)}%):{history['var_estimate'][a][-1]:.5f}"
                for a in alphas
            )
            print(f"  Epoch [{epoch+1}/{n_epochs}]  "
                  f"D:{history['d_loss'][-1]:.4f}  "
                  f"G:{history['g_loss'][-1]:.4f}  {parts}")

    # denormalizing wrapper
    class _DenormGen:
        def __init__(self, g, m, s):
            self.gen = g
            self.mean = m
            self.std  = s
        def generate(self, n, device=None):
            raw = self.gen.generate(n, device)
            return (raw * self.std + self.mean).astype(np.float32)
        def eval(self):
            self.gen.eval()

    return _DenormGen(generator, data_mean, data_std), discriminator, history


# -----------------------------------------------------------------------------
#  Help functoins
# -----------------------------------------------------------------------------
def paired_quantiles(real, gen, n_q=300):
    """
    Return (q_real, q_gen) at n_q probability levels uniformly in (0,1).
    Used for Q-Q plots.
    """
    probs  = np.linspace(0.5 / n_q, 1 - 0.5 / n_q, n_q)
    q_real = np.quantile(real.flatten(), probs)
    q_gen  = np.quantile(gen.flatten(),  probs)
    return q_real, q_gen


def worst_windows(data, n=5):
    """
    Return the n windows whose minimum return is most negative,
    i.e. the crash-like scenarios inside the dataset.
    """
    min_per_window = data.min(axis=1)          # shape (n_windows,)
    idx = np.argsort(min_per_window)[:n]       # ascending → most negative first
    return data[idx]


def _shared_bins(real, n_bins=70, tail_factor=1.6):
    """Histogram bins centred on the real data, slightly extended."""
    p01, p99 = np.percentile(real.flatten(), [0.5, 99.5])
    lo = p01 * tail_factor if p01 < 0 else p01 / tail_factor
    hi = p99 * tail_factor if p99 > 0 else p99 / tail_factor
    return np.linspace(lo, hi, n_bins)


# -----------------------------------------------------------------------------
#  Plot 1–5: per-model histogram + Q-Q  (1 × 2 figures)
# -----------------------------------------------------------------------------
def _fill_hist_qq(ax_h, ax_q, real, gen, gen_name, gen_color, bins):
    """Populate a (histogram, Q-Q) pair of axes."""
    real_f = real.flatten()
    gen_f  = gen.flatten()

    #  histogram
    ax_h.hist(real_f, bins=bins, density=True, alpha=0.55,
              color=C['real'], label='Real  N(0, 0.02²)')
    ax_h.hist(gen_f,  bins=bins, density=True, alpha=0.55,
              color=gen_color, label=gen_name)

    # Overlay theoretical N(0, 0.02²) density for reference
    x_pdf = np.linspace(bins[0], bins[-1], 400)
    ax_h.plot(x_pdf, stats.norm.pdf(x_pdf, 0, 0.02),
              'k--', linewidth=1.2, alpha=0.6, label='N(0, 0.02²) pdf')

    ax_h.set_xlabel('Daily log-return')
    ax_h.set_ylabel('Density')
    ax_h.set_title(f'Distribution – Real vs {gen_name}')
    ax_h.legend()

    # Q-Q plot
    qr, qg = paired_quantiles(real, gen)
    lo = min(qr.min(), qg.min()) * 1.08
    hi = max(qr.max(), qg.max()) * 1.08
    ax_q.scatter(qr, qg, s=6, alpha=0.55, color=gen_color)
    ax_q.plot([lo, hi], [lo, hi], 'k--', linewidth=1.1,
              label='y = x  (perfect fit)')
    ax_q.set_xlim(lo, hi)
    ax_q.set_ylim(lo, hi)
    ax_q.set_xlabel('Real quantiles')
    ax_q.set_ylabel(f'{gen_name} quantiles')
    ax_q.set_title(f'Q–Q Plot – Real vs {gen_name}')
    ax_q.legend()
    ax_q.set_aspect('equal', adjustable='box')


def save_hist_qq(real, gen, gen_name, gen_color, filename):
    """Save a 1×2 (histogram | Q-Q) figure to OUTPUT_DIR."""
    bins = _shared_bins(real)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'Synthetic Gaussian data  ·  Real vs {gen_name}',
        fontsize=13, y=1.01,
    )
    _fill_hist_qq(axes[0], axes[1], real, gen, gen_name, gen_color, bins)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {path}')


# -----------------------------------------------------------------------------
#  Plot 6: left-tail comparison  (1 × 4)
# -----------------------------------------------------------------------------
def save_tail_comparison(real, wgan, tailgan, egan,
                         alpha=0.05, worst_pct=15):
    """
    Four-panel figure showing the left tail (worst worst_pct %) of
    each dataset.  VaR(alpha) and ES(alpha) are marked as vertical lines.
    """
    datasets = [
        (real,    'Real Data',      C['real']),
        (wgan,    'WGAN-GP',        C['wgan']),
        (tailgan, 'Tail-GAN',       C['tailgan']),
        (egan,    'Expectile-GAN',  C['expectile']),
    ]

    # Bin range driven by real data
    tail_min = float(np.percentile(real.flatten(), 0.1))
    tail_max = float(np.percentile(real.flatten(), worst_pct))
    bins = np.linspace(tail_min, tail_max, 45)

    ymax = 0.0
    for d, _, _ in datasets:
        flat = d.flatten()
        thr  = np.percentile(flat, worst_pct)
        h, _ = np.histogram(flat[flat <= thr], bins=bins, density=True)
        if h.size:
            ymax = max(ymax, h.max())
    ymax *= 1.12

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f'Left-Tail Comparison  '
        f'(worst {worst_pct} % of returns  ·  '
        f'VaR and ES at alpha = {int(alpha * 100)} %)',
        fontsize=13,
    )

    for ax, (data_arr, name, color) in zip(axes, datasets):
        flat = data_arr.flatten()
        thr  = np.percentile(flat, worst_pct)
        tail = flat[flat <= thr]

        var = calculate_var(data_arr, alpha)
        es  = calculate_es(data_arr, alpha)

        ax.hist(tail, bins=bins, density=True, alpha=0.70,
                color=color, label=name)
        ax.axvline(var, color='crimson',    linestyle='--', linewidth=2.0,
                   label=f'VaR({int(alpha*100)}%) = {var:.4f}')
        ax.axvline(es,  color='darkorange', linestyle='--', linewidth=2.0,
                   label=f'ES({int(alpha*100)}%)  = {es:.4f}')

        ax.set_xlim(tail_min, tail_max)
        ax.set_ylim(0, ymax)
        ax.set_title(name)
        ax.set_xlabel('Daily log-return')
        ax.legend(fontsize=7)

    axes[0].set_ylabel('Density')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'tail_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {path}')


# -----------------------------------------------------------------------------
#  Plot 7: training curves  (2 × 3)
# -----------------------------------------------------------------------------
def save_training_curves(wgan_hist, tailgan_hist, egan_hist,
                         alphas, tau_levels):
    """
    2 × 3 figure:
      Row 0  –  generator & discriminator/critic losses for each model
      Row 1  –  WGAN-GP Wasserstein estimate | Tail-GAN VaR/ES per alpha |
                Expectile-GAN expectile estimates per tau
    """
    # Color for the multi-alpha / multi-tau panels, looks nice!
    risk_colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    ls_solid    = ['-',  '--', ':',  '-.']
    ls_dashed   = ['-.', ':',  '--', '-' ]

    fig = plt.figure(figsize=(19, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    fig.suptitle(
        'Training Curves  –  WGAN-GP  ·  Tail-GAN  ·  Expectile-GAN',
        fontsize=14,
    )

    epochs = np.arange(1, len(wgan_hist['critic_loss']) + 1)

    # Row 0: losses
    ax = axes[0][0]
    ax.plot(epochs, wgan_hist['critic_loss'],    label='Critic',    alpha=0.85)
    ax.plot(epochs, wgan_hist['generator_loss'], label='Generator', alpha=0.85)
    ax.set_title('WGAN-GP – Training Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    ax = axes[0][1]
    ax.plot(epochs, tailgan_hist['d_loss'], label='Discriminator', alpha=0.85)
    ax.plot(epochs, tailgan_hist['g_loss'], label='Generator',     alpha=0.85)
    ax.set_title('Tail-GAN – Training Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    ax = axes[0][2]
    ax.plot(epochs, egan_hist['d_loss'], label='Discriminator', alpha=0.85)
    ax.plot(epochs, egan_hist['g_loss'], label='Generator',     alpha=0.85)
    ax.set_title('Expectile-GAN – Training Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # Row 1, panel 0: WGAN-GP Wasserstein distance
    ax = axes[1][0]
    ax.plot(epochs, wgan_hist['wasserstein_dist'],
            color='steelblue', alpha=0.85)
    ax.set_title('WGAN-GP – Wasserstein Distance $\\hat{W}$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$\\hat{W}$')

    # Row 1, panel 1: Tail-GAN VaR and ES per alpha
    ax = axes[1][1]
    var_est = tailgan_hist['var_estimate']
    es_est  = tailgan_hist['es_estimate']
    t_epochs = np.arange(1, len(tailgan_hist['d_loss']) + 1)

    for i, a in enumerate(alphas):
        col = risk_colors[i % len(risk_colors)]
        ls1 = ls_solid[i  % len(ls_solid)]
        ls2 = ls_dashed[i % len(ls_dashed)]
        if isinstance(var_est, dict) and a in var_est:
            ax.plot(t_epochs, var_est[a], color=col, linestyle=ls1,
                    linewidth=1.5, alpha=0.90,
                    label=f'VaR  alpha={a}')
            ax.plot(t_epochs, es_est[a],  color=col, linestyle=ls2,
                    linewidth=1.5, alpha=0.60,
                    label=f'ES  alpha={a}')
        else:
            # Fallback: history is a flat list (single-alpha old format)
            ax.plot(t_epochs, var_est, label=f'VaR alpha={alphas[0]}',
                    alpha=0.90)
            ax.plot(t_epochs, es_est,  label=f'ES  alpha={alphas[0]}',
                    alpha=0.60, linestyle='--')
            break

    ax.set_title('Tail-GAN – VaR and ES Estimate Tracking')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Estimated value')
    ax.legend(fontsize=7.5)

    # Row 1, panel 2: Expectile-GAN expectile estimates per tau
    ax = axes[1][2]
    e_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    e_epochs = np.arange(1, len(egan_hist['d_loss']) + 1)
    for i, tau in enumerate(tau_levels):
        ax.plot(e_epochs,
                egan_hist['expectile_estimates'][tau],
                color=e_colors[i % len(e_colors)],
                linewidth=1.5, alpha=0.88,
                label=f'tau = {tau}')

    ax.set_title('Expectile-GAN – Expectile Estimate Tracking')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Estimated expectile')
    ax.legend(fontsize=7.5)

    path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {path}')


# -----------------------------------------------------------------------------
#  Plot 8: random sample paths  (4 stacked panels)
# -----------------------------------------------------------------------------
def save_sample_paths(real, wgan, tailgan, egan, n_paths=5):
    """
    Four vertically stacked subplots, each showing n_paths randomly chosen
    return windows from the corresponding dataset.
    """
    datasets = [
        (real,    'Real Data  N(0, 0.02²)', C['real']),
        (wgan,    'WGAN-GP',                C['wgan']),
        (tailgan, 'Tail-GAN  (alpha=[0.01, 0.05])',       C['tailgan']),
        (egan,    'Expectile-GAN  (tau=[0.01, 0.05, 0.10])', C['expectile']),
    ]
    ylim = 0.09

    fig, axes = plt.subplots(4, 1, figsize=(13, 15),
                             sharex=True, sharey=True)
    fig.suptitle(f'{n_paths} Random Sample Return Paths', fontsize=14)

    rng = np.random.default_rng(seed=7)
    T   = real.shape[1]
    x   = np.arange(T)

    for ax, (data_arr, name, color) in zip(axes, datasets):
        idx = rng.choice(data_arr.shape[0], n_paths, replace=False)
        for i in idx:
            ax.plot(x, data_arr[i], color=color, alpha=0.75, linewidth=0.85)
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_title(name, fontsize=11)
        ax.set_ylabel('Return', fontsize=9)
        ax.set_ylim(-ylim, ylim)

    axes[-1].set_xlabel('Time step  (days)', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'sample_paths.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {path}')


# -----------------------------------------------------------------------------
#  Plot 9: worst-case (crash) paths  (4 stacked panels)
# -----------------------------------------------------------------------------
def save_sample_losses_paths(real, wgan, tailgan, egan, n_paths=5):
    """
    Four vertically stacked subplots.  For each dataset the n_paths windows
    with the most negative single-step return are selected, showing the
    model's ability (or inability) to generate crisis-like paths.
    """
    datasets = [
        (real,    'Real Data  N(0, 0.02²)',            C['real']),
        (wgan,    'WGAN-GP',                           C['wgan']),
        (tailgan, 'Tail-GAN  (alpha=[0.01, 0.05])',        C['tailgan']),
        (egan,    'Expectile-GAN  (tau=[0.01, 0.05, 0.10])', C['expectile']),
    ]
    ylim = max(float(np.abs(d).max()) for d, _, _ in datasets) * 1.08

    fig, axes = plt.subplots(4, 1, figsize=(13, 15),
                             sharex=True, sharey=True)
    fig.suptitle(
        f'{n_paths} Worst-Case Return Paths'
        '  (windows with most negative minimum return)',
        fontsize=13,
    )

    T = real.shape[1]
    x = np.arange(T)

    for ax, (data_arr, name, color) in zip(axes, datasets):
        crash_windows = worst_windows(data_arr, n=n_paths)
        for i in range(n_paths):
            ax.plot(x, crash_windows[i],
                    color=color, alpha=0.80, linewidth=0.85)
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

        # Mark the global minimum in each path
        for i in range(n_paths):
            t_min = int(np.argmin(crash_windows[i]))
            v_min = float(crash_windows[i, t_min])
            ax.scatter(t_min, v_min, color='black', s=22, zorder=5)

        ax.set_title(name, fontsize=11)
        ax.set_ylabel('Return', fontsize=9)
        ax.set_ylim(-ylim, ylim)

    axes[-1].set_xlabel('Time step  (days)', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'sample_losses_paths.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {path}')


# -----------------------------------------------------------------------------
#  Print a numerical summary table
# -----------------------------------------------------------------------------
def print_summary(real, wgan, tailgan_s, tailgan_m, egan_s, egan_m,
                  alphas, tau_levels):
    hdr = (f"{'Metric':<30} {'Real':>10} {'WGAN-GP':>10} "
           f"{'TG alpha=0.05':>10} {'TG multi':>10} "
           f"{'EG tau=0.1':>10} {'EG multi':>10}")
    print('\n' + '=' * len(hdr))
    print('Numerical Summary – Synthetic Gaussian Study')
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))

    datasets = [real, wgan, tailgan_s, tailgan_m, egan_s, egan_m]
    names    = ['Real', 'WGAN', 'TG-s', 'TG-m', 'EG-s', 'EG-m']

    def row(label, values):
        vals_str = ''.join(f'{v:>10.5f}' for v in values)
        print(f'{label:<30}{vals_str}')

    row('Mean',    [d.mean()   for d in datasets])
    row('Std Dev', [d.std()    for d in datasets])
    row('Skewness',[float(stats.skew(d.flatten())) for d in datasets])
    row('Kurtosis',[float(stats.kurtosis(d.flatten())) for d in datasets])

    for alpha in alphas:
        row(f'VaR ({int(alpha*100)}%)',
            [calculate_var(d, alpha) for d in datasets])
        row(f'ES  ({int(alpha*100)}%)',
            [calculate_es(d, alpha)  for d in datasets])

    for tau in tau_levels:
        row(f'Expectile tau={tau}',
            [compute_expectile(d, tau) for d in datasets])

    print('=' * len(hdr) + '\n')


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------
def main(n_epochs: int = 500):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 70)
    print('  gaussian_comparison.py')
    print(f'  Output folder : {OUTPUT_DIR}/')
    print(f'  Training epochs: {n_epochs}')
    print('=' * 70)

    # shared settings
    ALPHAS     = [0.01, 0.05]
    TAU_LEVELS = [0.01, 0.05, 0.10]
    N_GEN      = 1000           # samples generated for evaluation

    # data
    data = make_gaussian(n=1000, T=50, std=0.02)
    print(f'\nData  shape={data.shape}  '
          f'mean={data.mean():.6f}  std={data.std():.6f}\n')

    # 1/5  WGAN-GP
    print('─' * 60)
    print('1 / 5   WGAN-GP')
    print('─' * 60)
    wgan_gen, wgan_hist = train_wgan_gp(
        data, n_epochs=n_epochs, batch_size=64, latent_dim=100,
        lr=1e-4, n_critic=5, lambda_gp=10,
        beta1=0.0, beta2=0.9, verbose=True,
    )
    wgan_samples = wgan_gen.generate(N_GEN).astype(np.float32)

    # 2/5  Tail-GAN  single alpha = 0.05
    print('\n' + '─' * 60)
    print('2 / 5   Tail-GAN  (single  alpha = 0.05)')
    print('─' * 60)
    tailgan_s_gen, _, _ = train_tailgan_tracked(
        data, n_epochs=n_epochs, alphas=[0.05], verbose=True,
    )
    tailgan_s_samples = tailgan_s_gen.generate(N_GEN)

    # 3/5  Tail-GAN  multi-alpha = [0.01, 0.05]
    print('\n' + '─' * 60)
    print('3 / 5   Tail-GAN  (multi   alpha = [0.01, 0.05])')
    print('─' * 60)
    tailgan_m_gen, _, tailgan_hist = train_tailgan_tracked(
        data, n_epochs=n_epochs, alphas=ALPHAS, verbose=True,
    )
    tailgan_m_samples = tailgan_m_gen.generate(N_GEN)

    # 4/5  Expectile-GAN  single tau = 0.10
    print('\n' + '─' * 60)
    print('4 / 5   Expectile-GAN  (single  tau = 0.10)')
    print('─' * 60)
    egan_s_gen, _, _ = train_expectile_gan(
        data, n_epochs=n_epochs, batch_size=64, latent_dim=100,
        lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
        tau_levels=[0.10], verbose=True,
    )
    egan_s_samples = egan_s_gen.generate(N_GEN).astype(np.float32)

    # 5/5  Expectile-GAN  multi-tau = [0.01, 0.05, 0.10]
    print('\n' + '─' * 60)
    print('5 / 5   Expectile-GAN  (multi  tau = [0.01, 0.05, 0.10])')
    print('─' * 60)
    egan_m_gen, _, egan_hist = train_expectile_gan(
        data, n_epochs=n_epochs, batch_size=64, latent_dim=100,
        lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
        tau_levels=TAU_LEVELS, verbose=True,
    )
    egan_m_samples = egan_m_gen.generate(N_GEN).astype(np.float32)

    # figures
    print('\n' + '=' * 70)
    print('  Saving figures ...')
    print('=' * 70)

    # 1 – WGAN-GP histogram + Q-Q
    save_hist_qq(
        data, wgan_samples,
        gen_name='WGAN-GP',
        gen_color=C['wgan'],
        filename='gaussian_WGAN',
    )

    # 2 – Tail-GAN single alpha
    save_hist_qq(
        data, tailgan_s_samples,
        gen_name='Tail-GAN  (alpha = 0.05)',
        gen_color=C['tailgan'],
        filename='gaussian_TAILGANalpha',
    )

    # 3 – Tail-GAN multi-alpha
    save_hist_qq(
        data, tailgan_m_samples,
        gen_name='Tail-GAN  (alpha ∈ {0.01, 0.05})',
        gen_color=C['tailgan'],
        filename='gaussian_TAILGANalphas',
    )

    # 4 – Expectile-GAN single tau
    save_hist_qq(
        data, egan_s_samples,
        gen_name='Expectile-GAN  (tau = 0.10)',
        gen_color=C['expectile'],
        filename='gaussian_EXPECTILEGANtau',
    )

    # 5 – Expectile-GAN multi-tau
    save_hist_qq(
        data, egan_m_samples,
        gen_name='Expectile-GAN  (tau ∈ {0.01, 0.05, 0.10})',
        gen_color=C['expectile'],
        filename='gaussian_EXPECTILEGANtaus',
    )

    # 6 – left-tail comparison  (uses multi-alpha Tail-GAN and multi-tau EGAN)
    save_tail_comparison(
        data, wgan_samples, tailgan_m_samples, egan_m_samples,
        alpha=0.05, worst_pct=15,
    )

    # 7 – training curves  (uses multi-alpha Tail-GAN and multi-tau EGAN hists)
    save_training_curves(
        wgan_hist, tailgan_hist, egan_hist,
        alphas=ALPHAS,
        tau_levels=TAU_LEVELS,
    )

    # 8 – random sample paths  (multi-alpha / multi-tau models)
    save_sample_paths(
        data, wgan_samples, tailgan_m_samples, egan_m_samples,
        n_paths=5,
    )

    # 9 – worst-case (crash) paths  (multi-alpha / multi-tau models)
    save_sample_losses_paths(
        data, wgan_samples, tailgan_m_samples, egan_m_samples,
        n_paths=5,
    )

    # numerical summary
    print_summary(
        data, wgan_samples,
        tailgan_s_samples, tailgan_m_samples,
        egan_s_samples,    egan_m_samples,
        alphas=ALPHAS, tau_levels=TAU_LEVELS,
    )

    print('=' * 70)
    print(f'  All done.  9 figures saved in  {OUTPUT_DIR}/')
    print('=' * 70)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gaussian comparison figures for the thesis.'
    )
    parser.add_argument(
        '--epochs', type=int, default=500,
        help='Training epochs for each model  (default: 500; use 50 for a quick test)',
    )
    args = parser.parse_args()
    main(n_epochs=args.epochs)
