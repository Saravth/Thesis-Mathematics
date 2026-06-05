"""
Generates all comparison figures for the S&P 500 study
Creates a folder  sp500_comparison/  containing:

  sp500_WGAN.png                 histogram + Q-Q: WGAN-GP vs real S&P 500
  sp500_TAILGAN.png              histogram + Q-Q: Tail-GAN (alpha=[0.01,0.05]) vs real
  sp500_EXPECTILEGAN.png         histogram + Q-Q: Expectile-GAN (tau=[0.01,0.05,0.10]) vs real
  sp500_tail_comparison.png      worst 15 % left tail across all models
  sp500_training_curves.png      2 × 3 loss and risk-estimate tracking
  sp500_sample_paths.png         5 random return paths per model (4 stacked panels)
  sp500_sample_losses_paths.png  5 worst-case return paths per model (4 stacked panels)

S&P 500 data (2016-01-01 → 2023-12-31) is downloaded via yfinance and cached
locally as  sp500_cache.npy  so subsequent runs are instant even without internet.
If yfinance is unavailable you can supply a pre-downloaded file with --data.
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

# GAN modules
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

# global matplotlib style
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
OUTPUT_DIR  = 'sp500_comparison'
CACHE_FILE  = 'sp500_cache.npy'           # 1-D array of log-returns
WINDOW_SIZE = 50                          # rolling window length  T
N_GEN       = 2000                        # generated samples for evaluation

C = {
    'real':      '#1f77b4',
    'wgan':      '#ff7f0e',
    'tailgan':   '#2ca02c',
    'expectile': '#d62728',
}

# -----------------------------------------------------------------------------
#  S&P 500 data
# -----------------------------------------------------------------------------
def _try_import_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        pass
    try:
        import subprocess
        subprocess.run(
            ['pip', 'install', 'yfinance', '--quiet'],
            check=True,
        )
        import yfinance as yf
        return yf
    except Exception:
        return None


def download_sp500(start='2016-01-01', end='2023-12-31',
                   cache_path=CACHE_FILE):
    """
    Download S&P 500 daily log-returns and cache them as a .npy file.
    If the cache already exists it is loaded directly (no internet needed).

    Returns
    -------
    log_returns : np.ndarray, shape (n_days,), dtype float32
    """
    if os.path.isfile(cache_path):
        print(f'  Loading cached S&P 500 returns from  {cache_path}')
        returns = np.load(cache_path).astype(np.float32)
        print(f'  {len(returns)} daily log-returns  '
              f'(mean={returns.mean():.5f}, std={returns.std():.5f})')
        return returns

    yf = _try_import_yfinance()
    if yf is None:
        raise RuntimeError(
            'yfinance is not installed and could not be installed automatically.\n'
            'Install it with  pip install yfinance  or supply a cached .npy\n'
            'file via  --data <path>.'
        )

    print(f'  Downloading S&P 500 (^GSPC) from {start} to {end} …')
    ticker = yf.Ticker('^GSPC')
    df     = ticker.history(start=start, end=end)
    if df.empty:
        raise RuntimeError('yfinance returned an empty DataFrame. '
                           'Check your internet connection.')

    prices  = df['Close'].dropna().values
    returns = np.diff(np.log(prices)).astype(np.float32)
    np.save(cache_path, returns)
    print(f'  Downloaded {len(returns)} daily log-returns → cached to {cache_path}')
    print(f'  mean={returns.mean():.5f}  std={returns.std():.5f}  '
          f'min={returns.min():.5f}  max={returns.max():.5f}')
    return returns


def create_rolling_windows(returns, T=WINDOW_SIZE):
    """
    Slice a 1-D return array into overlapping windows of length T.

    Returns
    -------
    windows : np.ndarray, shape (n_windows, T), dtype float32
    """
    n_windows = len(returns) - T + 1
    windows   = np.array(
        [returns[i: i + T] for i in range(n_windows)],
        dtype=np.float32,
    )
    print(f'  Rolling windows: {n_windows} × {T}  '
          f'(mean={windows.mean():.5f}, std={windows.std():.5f})')
    return windows


# -----------------------------------------------------------------------------
#  Extended Tail-GAN training with per-alpha history.
# -----------------------------------------------------------------------------
def train_tailgan_tracked(data, n_epochs=500, batch_size=64, latent_dim=100,
                          lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
                          alphas=(0.01, 0.05), verbose=True):
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
                          alphas=(0.01, 0.05), verbose=True):
    """
    Train Tail-GAN and return a history dict in which
    var_estimate and es_estimate are  {alpha: [epoch_means]}  dicts,
    enabling per-alpha visualisation in the training-curves figure.
    """
    device     = get_device()
    alphas     = list(alphas)
    output_dim = data.shape[1]

    data_mean = float(data.mean())
    data_std  = float(data.std())
    data_norm = (data - data_mean) / data_std

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

            opt_D.zero_grad()
            z        = torch.randn(bs, latent_dim, device=device)
            fake     = generator(z)
            fake_est = discriminator(fake.detach())
            real_est = discriminator(real)
            d_loss   = -(scorer(fake_est, real)
                         - lambda_dual * scorer(real_est, real))
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            opt_D.step()
            d_buf.append(d_loss.item())

            opt_G.zero_grad()
            z2    = torch.randn(bs, latent_dim, device=device)
            fake2 = generator(z2)
            g_loss = scorer(discriminator(fake2), real)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_G.step()
            g_buf.append(g_loss.item())

            with torch.no_grad():
                for i, a in enumerate(alphas):
                    v = (fake_est[:, 2 * i    ].mean().item()
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
            parts = '  '.join(
                f"VaR({int(a*100)}%):{history['var_estimate'][a][-1]:.5f}"
                for a in alphas
            )
            print(f'  [{epoch+1}/{n_epochs}]  '
                  f'D:{history["d_loss"][-1]:.4f}  '
                  f'G:{history["g_loss"][-1]:.4f}  {parts}')

    class _DenormGen:
        def __init__(self, g, m, s):
            self.gen = g; self.mean = m; self.std = s
        def generate(self, n, device=None):
            raw = self.gen.generate(n, device)
            return (raw * self.std + self.mean).astype(np.float32)
        def eval(self): self.gen.eval()

    return _DenormGen(generator, data_mean, data_std), discriminator, history


# -----------------------------------------------------------------------------
#  Help functions
# -----------------------------------------------------------------------------
def paired_quantiles(real, gen, n_q=300):
    probs  = np.linspace(0.5 / n_q, 1 - 0.5 / n_q, n_q)
    q_real = np.quantile(real.flatten(), probs)
    q_gen  = np.quantile(gen.flatten(),  probs)
    return q_real, q_gen


def worst_windows(data, n=5):
    idx = np.argsort(data.min(axis=1))[:n]
    return data[idx]


def _shared_bins(real, n_bins=75):
    lo = np.percentile(real.flatten(), 0.1)
    hi = np.percentile(real.flatten(), 99.9)
    half = max(abs(lo), abs(hi))
    return np.linspace(-half * 1.8, half * 1.8, n_bins)


# -----------------------------------------------------------------------------
#  Plot 1–3: histogram + Q-Q  (1 × 2)
# -----------------------------------------------------------------------------
def _fill_hist_qq_sp500(ax_h, ax_q, real, gen, gen_name, gen_color, bins):
    """
    Populate a (histogram, Q-Q) pair of axes for the S&P 500 study.

    A fitted normal distribution N(mu_real, sigma_real²) is overlaid as a dashed
    black curve to highlight the heavy tails of the empirical distribution.
    """
    real_f = real.flatten()
    gen_f  = gen.flatten()
    mu_r, sigma_r = float(real_f.mean()), float(real_f.std())

    # histogram
    ax_h.hist(real_f, bins=bins, density=True, alpha=0.55,
              color=C['real'], label='Real  S&P 500')
    ax_h.hist(gen_f,  bins=bins, density=True, alpha=0.55,
              color=gen_color, label=gen_name)

    # Fitted normal reference
    x_pdf = np.linspace(bins[0], bins[-1], 500)
    ax_h.plot(x_pdf,
              stats.norm.pdf(x_pdf, mu_r, sigma_r),
              'k--', linewidth=1.3, alpha=0.65,
              label=f'Fitted N({mu_r:.4f}, {sigma_r:.4f}²)')

    ax_h.set_xlabel('Daily log-return')
    ax_h.set_ylabel('Density')
    ax_h.set_title(f'Distribution  –  Real S&P 500 vs {gen_name}')
    ax_h.legend()

    # Q-Q
    qr, qg = paired_quantiles(real, gen)
    lo = min(qr.min(), qg.min()) * 1.08
    hi = max(qr.max(), qg.max()) * 1.08
    ax_q.scatter(qr, qg, s=6, alpha=0.55, color=gen_color)
    ax_q.plot([lo, hi], [lo, hi], 'k--', linewidth=1.1, label='y = x')
    ax_q.set_xlim(lo, hi); ax_q.set_ylim(lo, hi)
    ax_q.set_xlabel('Real S&P 500 quantiles')
    ax_q.set_ylabel(f'{gen_name} quantiles')
    ax_q.set_title(f'Q–Q Plot  –  Real S&P 500 vs {gen_name}')
    ax_q.legend()
    ax_q.set_aspect('equal', adjustable='box')


def save_hist_qq(real, gen, gen_name, gen_color, filename):
    bins = _shared_bins(real)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'S&P 500 Daily Log-Returns  ·  Real vs {gen_name}',
        fontsize=13, y=1.01,
    )
    _fill_hist_qq_sp500(axes[0], axes[1], real, gen,
                        gen_name, gen_color, bins)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {path}')

# -----------------------------------------------------------------------------
#  Plot 4: left-tail comparison  (1 × 4)
# -----------------------------------------------------------------------------
def save_tail_comparison(real, wgan, tailgan, egan,
                         alpha=0.05, worst_pct=15):
    datasets = [
        (real,    'Real S&P 500',   C['real']),
        (wgan,    'WGAN-GP',        C['wgan']),
        (tailgan, 'Tail-GAN',       C['tailgan']),
        (egan,    'Expectile-GAN',  C['expectile']),
    ]

    tail_min = float(np.percentile(real.flatten(), 0.05))
    tail_max = float(np.percentile(real.flatten(), worst_pct))
    bins = np.linspace(tail_min, tail_max, 50)

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
        f'S&P 500  –  Left-Tail Comparison  '
        f'(worst {worst_pct} %  ·  '
        f'VaR and ES at alpha = {int(alpha*100)} %)',
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
    path = os.path.join(OUTPUT_DIR, 'sp500_tail_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {path}')


# -----------------------------------------------------------------------------
#  Plot 5: training curves  (2 × 3)
# -----------------------------------------------------------------------------
def save_training_curves(wgan_hist, tailgan_hist, egan_hist,
                         alphas, tau_levels):
    risk_colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    ls_solid    = ['-',  '--', ':',  '-.']
    ls_dashed   = ['-.', ':',  '--', '-' ]

    fig = plt.figure(figsize=(19, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    fig.suptitle(
        'S&P 500  –  Training Curves: WGAN-GP  ·  Tail-GAN  ·  Expectile-GAN',
        fontsize=14,
    )

    w_epochs = np.arange(1, len(wgan_hist['critic_loss']) + 1)
    t_epochs = np.arange(1, len(tailgan_hist['d_loss']) + 1)
    e_epochs = np.arange(1, len(egan_hist['d_loss']) + 1)

    # Row 0: losses
    ax = axes[0][0]
    ax.plot(w_epochs, wgan_hist['critic_loss'],    label='Critic',    alpha=0.85)
    ax.plot(w_epochs, wgan_hist['generator_loss'], label='Generator', alpha=0.85)
    ax.set_title('WGAN-GP – Training Losses')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()

    ax = axes[0][1]
    ax.plot(t_epochs, tailgan_hist['d_loss'], label='Discriminator', alpha=0.85)
    ax.plot(t_epochs, tailgan_hist['g_loss'], label='Generator',     alpha=0.85)
    ax.set_title('Tail-GAN – Training Losses')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()

    ax = axes[0][2]
    ax.plot(e_epochs, egan_hist['d_loss'], label='Discriminator', alpha=0.85)
    ax.plot(e_epochs, egan_hist['g_loss'], label='Generator',     alpha=0.85)
    ax.set_title('Expectile-GAN – Training Losses')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()

    # Row 1: risk / metric estimates
    ax = axes[1][0]
    ax.plot(w_epochs, wgan_hist['wasserstein_dist'],
            color='steelblue', alpha=0.85)
    ax.set_title('WGAN-GP – Wasserstein Distance $\\hat{W}$')
    ax.set_xlabel('Epoch'); ax.set_ylabel('$\\hat{W}$')

    ax = axes[1][1]
    var_est = tailgan_hist['var_estimate']
    es_est  = tailgan_hist['es_estimate']
    for i, a in enumerate(alphas):
        col = risk_colors[i % len(risk_colors)]
        ls1 = ls_solid[i  % len(ls_solid)]
        ls2 = ls_dashed[i % len(ls_dashed)]
        if isinstance(var_est, dict) and a in var_est:
            ax.plot(t_epochs, var_est[a], color=col, linestyle=ls1,
                    linewidth=1.5, alpha=0.90, label=f'VaR  alpha={a}')
            ax.plot(t_epochs, es_est[a],  color=col, linestyle=ls2,
                    linewidth=1.5, alpha=0.60, label=f'ES   alpha={a}')
        else:
            ax.plot(t_epochs, var_est, label=f'VaR alpha={alphas[0]}', alpha=0.90)
            ax.plot(t_epochs, es_est,  label=f'ES  alpha={alphas[0]}', alpha=0.60,
                    linestyle='--')
            break
    ax.set_title('Tail-GAN – VaR and ES Estimate Tracking')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Estimated value'); ax.legend(fontsize=7.5)

    ax = axes[1][2]
    e_cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, tau in enumerate(tau_levels):
        ax.plot(e_epochs,
                egan_hist['expectile_estimates'][tau],
                color=e_cols[i % len(e_cols)],
                linewidth=1.5, alpha=0.88,
                label=f'tau = {tau}')
    ax.set_title('Expectile-GAN – Expectile Estimate Tracking')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Estimated expectile')
    ax.legend(fontsize=7.5)

    path = os.path.join(OUTPUT_DIR, 'sp500_training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {path}')


# -----------------------------------------------------------------------------
#  Plot 6: random sample paths  (4 stacked panels)
# -----------------------------------------------------------------------------
def save_sample_paths(real, wgan, tailgan, egan, n_paths=5):
    datasets = [
        (real,    'Real S&P 500',                          C['real']),
        (wgan,    'WGAN-GP',                               C['wgan']),
        (tailgan, 'Tail-GAN  (alpha=[0.01, 0.05])',            C['tailgan']),
        (egan,    'Expectile-GAN  (tau=[0.01, 0.05, 0.10])', C['expectile']),
    ]

    ylim = 0.1 # looks ok

    fig, axes = plt.subplots(4, 1, figsize=(13, 15),
                             sharex=True, sharey=True)
    fig.suptitle(
        f'S&P 500  –  {n_paths} Random 50-Day Return Windows',
        fontsize=14,
    )

    rng = np.random.default_rng(seed=7)
    x   = np.arange(WINDOW_SIZE)

    for ax, (data_arr, name, color) in zip(axes, datasets):
        idx = rng.choice(data_arr.shape[0], n_paths, replace=False)
        for i in idx:
            ax.plot(x, data_arr[i], color=color, alpha=0.75, linewidth=0.85)
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_title(name, fontsize=11)
        ax.set_ylabel('Return', fontsize=9)
        ax.set_ylim(-ylim, ylim)

    axes[-1].set_xlabel('Time step  (trading days within window)', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'sp500_sample_paths.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {path}')


# -----------------------------------------------------------------------------
#  Plot 7: worst-case (crash) paths  (4 stacked panels)
# -----------------------------------------------------------------------------
def save_sample_losses_paths(real, wgan, tailgan, egan, n_paths=5):
    """
    Each panel shows the n_paths windows with the most negative single-day
    return, i.e. the most crash-like scenarios in each dataset.

    """
    datasets = [
        (real,    'Real S&P 500',                          C['real']),
        (wgan,    'WGAN-GP',                               C['wgan']),
        (tailgan, 'Tail-GAN  (alpha=[0.01, 0.05])',            C['tailgan']),
        (egan,    'Expectile-GAN  (tau=[0.01, 0.05, 0.10])', C['expectile']),
    ]

    ylim = max(float(np.abs(d).max()) for d, _, _ in datasets) * 1.10

    fig, axes = plt.subplots(4, 1, figsize=(13, 15),
                             sharex=True, sharey=True)
    fig.suptitle(
        f'S&P 500  –  {n_paths} Worst-Case 50-Day Windows\n'
        '(windows with most negative single-day return)',
        fontsize=13,
    )

    x = np.arange(WINDOW_SIZE)

    for ax, (data_arr, name, color) in zip(axes, datasets):
        crash = worst_windows(data_arr, n=n_paths)
        for i in range(n_paths):
            ax.plot(x, crash[i], color=color, alpha=0.80, linewidth=0.85)
            # Mark the day with the minimum return
            t_min = int(np.argmin(crash[i]))
            v_min = float(crash[i, t_min])
            ax.scatter(t_min, v_min, color='black', s=25, zorder=5)

        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_title(name, fontsize=11)
        ax.set_ylabel('Return', fontsize=9)
        ax.set_ylim(-ylim, ylim)

    axes[-1].set_xlabel('Time step  (trading days within window)', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'sp500_sample_losses_paths.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {path}')


# -----------------------------------------------------------------------------
#  Numerical summary table
# -----------------------------------------------------------------------------
def print_summary(real, wgan, tailgan, egan, alphas, tau_levels):
    hdr = (f"{'Metric':<30} {'Real S&P500':>13} "
           f"{'WGAN-GP':>10} {'Tail-GAN':>10} {'Exp-GAN':>10}")
    sep = '─' * len(hdr)
    print('\n' + '═' * len(hdr))
    print('Numerical Summary  –  S&P 500 Study')
    print('═' * len(hdr))
    print(hdr)
    print(sep)

    datasets = [real, wgan, tailgan, egan]

    def row(label, values):
        print(f'{label:<30}'
              + ''.join(f'{v:>13.6f}' for v in values))

    row('Mean',     [d.mean() for d in datasets])
    row('Std Dev',  [d.std()  for d in datasets])
    row('Skewness', [float(stats.skew(d.flatten())) for d in datasets])
    row('Kurtosis', [float(stats.kurtosis(d.flatten())) for d in datasets])

    print(sep)
    for alpha in alphas:
        row(f'VaR ({int(alpha*100)}%)',
            [calculate_var(d, alpha) for d in datasets])
        row(f'ES  ({int(alpha*100)}%)',
            [calculate_es(d,  alpha) for d in datasets])

    print(sep)
    for tau in tau_levels:
        row(f'Expectile  tau={tau}',
            [compute_expectile(d, tau) for d in datasets])

    # Relative errors w.r.t. real data
    print('\n' + sep)
    print(f"{'Relative error (%)':30} {'':13} "
          f"{'WGAN-GP':>10} {'Tail-GAN':>10} {'Exp-GAN':>10}")
    print(sep)
    for alpha in alphas:
        r_var = calculate_var(real, alpha)
        r_es  = calculate_es(real,  alpha)
        errs_var = [abs(calculate_var(d, alpha) - r_var) / abs(r_var) * 100
                    for d in [wgan, tailgan, egan]]
        errs_es  = [abs(calculate_es(d,  alpha) - r_es)  / abs(r_es)  * 100
                    for d in [wgan, tailgan, egan]]
        print(f'{"VaR ("+str(int(alpha*100))+"%) err":<30}'
              + f'{"":>13}'
              + ''.join(f'{e:>10.2f}%' for e in errs_var))
        print(f'{"ES  ("+str(int(alpha*100))+"%) err":<30}'
              + f'{"":>13}'
              + ''.join(f'{e:>10.2f}%' for e in errs_es))
    for tau in tau_levels:
        r_e    = compute_expectile(real, tau)
        errs_e = [abs(compute_expectile(d, tau) - r_e) / abs(r_e) * 100
                  for d in [wgan, tailgan, egan]]
        print(f'{"Expectile (tau="+str(tau)+") err":<30}'
              + f'{"":>13}'
              + ''.join(f'{e:>10.2f}%' for e in errs_e))
    print('═' * len(hdr) + '\n')


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------
def main(n_epochs: int = 500, data_path: str = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 70)
    print('  sp500_comparison.py')
    print(f'  Output folder  : {OUTPUT_DIR}/')
    print(f'  Training epochs: {n_epochs}')
    print('=' * 70)

    # shared settings
    ALPHAS     = [0.01, 0.05]
    TAU_LEVELS = [0.01, 0.05, 0.10]

    # data
    print('\n-- Loading S&P 500 data --------------------------------')
    if data_path is not None:
        print(f'  Loading returns from {data_path}')
        raw_returns = np.load(data_path).astype(np.float32).ravel()
        print(f'  {len(raw_returns)} daily log-returns loaded')
    else:
        raw_returns = download_sp500(
            start='2016-01-01', end='2023-12-31',
            cache_path=CACHE_FILE,
        )

    data = create_rolling_windows(raw_returns, T=WINDOW_SIZE)
    print(f'  Final data shape: {data.shape}\n')

    # 1/3  WGAN-GP
    print('─' * 60)
    print('1 / 3   WGAN-GP')
    print('─' * 60)
    wgan_gen, wgan_hist = train_wgan_gp(
        data, n_epochs=n_epochs, batch_size=64, latent_dim=100,
        lr=1e-4, n_critic=5, lambda_gp=10,
        beta1=0.0, beta2=0.9, verbose=True,
    )
    wgan_samples = wgan_gen.generate(N_GEN).astype(np.float32)

    # 2/3  Tail-GAN  (alpha=[0.01, 0.05])
    print('\n' + '─' * 60)
    print('2 / 3   Tail-GAN  (alpha = [0.01, 0.05])')
    print('─' * 60)
    tailgan_gen, _, tailgan_hist = train_tailgan_tracked(
        data, n_epochs=n_epochs, alphas=ALPHAS, verbose=True,
    )
    tailgan_samples = tailgan_gen.generate(N_GEN)

    # 3/3  Expectile-GAN  (tau=[0.01, 0.05, 0.10])
    print('\n' + '─' * 60)
    print('3 / 3   Expectile-GAN  (tau = [0.01, 0.05, 0.10])')
    print('─' * 60)
    egan_gen, _, egan_hist = train_expectile_gan(
        data, n_epochs=n_epochs, batch_size=64, latent_dim=100,
        lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
        tau_levels=TAU_LEVELS, verbose=True,
    )
    egan_samples = egan_gen.generate(N_GEN).astype(np.float32)

    # Figures
    print('\n' + '=' * 70)
    print('  Saving figures ...')
    print('=' * 70)

    # 1 – WGAN-GP
    save_hist_qq(
        data, wgan_samples,
        gen_name='WGAN-GP',
        gen_color=C['wgan'],
        filename='sp500_WGAN',
    )

    # 2 – Tail-GAN
    save_hist_qq(
        data, tailgan_samples,
        gen_name='Tail-GAN  (alpha ∈ {0.01, 0.05})',
        gen_color=C['tailgan'],
        filename='sp500_TAILGAN',
    )

    # 3 – Expectile-GAN
    save_hist_qq(
        data, egan_samples,
        gen_name='Expectile-GAN  (tau ∈ {0.01, 0.05, 0.10})',
        gen_color=C['expectile'],
        filename='sp500_EXPECTILEGAN',
    )

    # 4 – left-tail comparison
    save_tail_comparison(
        data, wgan_samples, tailgan_samples, egan_samples,
        alpha=0.05, worst_pct=15,
    )

    # 5 – training curves
    save_training_curves(
        wgan_hist, tailgan_hist, egan_hist,
        alphas=ALPHAS,
        tau_levels=TAU_LEVELS,
    )

    # 6 – random sample paths
    save_sample_paths(
        data, wgan_samples, tailgan_samples, egan_samples,
        n_paths=5,
    )

    # 7 – worst-case (crash) paths
    save_sample_losses_paths(
        data, wgan_samples, tailgan_samples, egan_samples,
        n_paths=5,
    )

    # numerical summary
    print_summary(
        data, wgan_samples, tailgan_samples, egan_samples,
        alphas=ALPHAS, tau_levels=TAU_LEVELS,
    )

    print('=' * 70)
    print(f'  All done.  7 figures saved in  {OUTPUT_DIR}/')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='S&P 500 comparison figures for the thesis.'
    )
    parser.add_argument(
        '--epochs', type=int, default=500,
        help='Training epochs for each model  (default: 500; use 300 for a quick test)',
    )
    parser.add_argument(
        '--data', type=str, default=None,
        metavar='PATH',
        help=('Path to a .npy file containing a 1-D array of daily log-returns.\n'
              'If omitted, data is downloaded via yfinance and cached as '
              f'{CACHE_FILE}.'),
    )
    args = parser.parse_args()
    main(n_epochs=args.epochs, data_path=args.data)
