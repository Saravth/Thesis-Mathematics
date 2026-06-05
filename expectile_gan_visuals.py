"""
Trains Expectile-GAN at a single tau level (tau=0.05) on:
  1. Synthetic Gaussian data:  N(0, 0.02^2), 1000 samples x 50 timesteps
  2. S&P 500 daily log-returns: rolling 50-day windows (2016-2023)

Produces figures and tables.
All figures show Expectile-GAN vs real data only (no WGAN-GP or Tail-GAN).

Output folder: expectile_gan_visuals/

Figures produced:
  expectile_gaussian.png          hist + Q-Q: real vs Expectile-GAN (Gaussian)
  expectile_sp500.png             hist + Q-Q: real vs Expectile-GAN (S&P 500)
  expectile_sp500_tail.png        left-tail comparison (real vs Expectile-GAN)
  expectile_sp500_sample_paths.png  5 random return windows (2 panels)
  expectile_sp500_crash_paths.png   5 worst-case windows (2 panels)
  expectile_gaussian_tail.png     left-tail Gaussian (real vs Expectile-GAN)
  expectile_sp500_stats.tex       LaTeX table with numerical statistics
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from expectile_gan import train_expectile_gan, compute_expectile
from tail_gan import calculate_var, calculate_es

# constants
DEFAULT_OUTPUT_DIR = 'expectile_gan_visuals'
CACHE_FILE         = 'sp500_cache.npy'
WINDOW_SIZE        = 50
N_GEN              = 2000
DEFAULT_TAU        = 0.05

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
TAU        = DEFAULT_TAU

C = {
    'real':      '#1f77b4',
    'expectile': '#d62728',
}

plt.rcParams.update({
    'font.size':       10,
    'axes.titlesize':  12,
    'axes.labelsize':  10,
    'legend.fontsize':  8,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'axes.grid':       True,
    'grid.alpha':      0.30,
    'grid.linestyle':  '--',
})

# data helpers
def generate_gaussian_data(n_samples=1000, n_timesteps=50, mean=0.0, std=0.02, seed=42):
    np.random.seed(seed)
    return np.random.normal(mean, std, (n_samples, n_timesteps)).astype(np.float32)


def load_sp500(cache_path=CACHE_FILE):
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f'{cache_path} not found. Run sp500_comparison.py first to create the cache.')
    returns = np.load(cache_path).astype(np.float32)
    print(f'  Loaded {len(returns)} S&P 500 daily log-returns from {cache_path}')
    return returns


def create_rolling_windows(returns, T=WINDOW_SIZE):
    n_windows = len(returns) - T + 1
    windows = np.array([returns[i: i + T] for i in range(n_windows)], dtype=np.float32)
    print(f'  Rolling windows: {n_windows} x {T}  (std={windows.std():.5f})')
    return windows


# bins
def _shared_bins(real, gen, n_bins=60):
    all_flat = np.concatenate([real.flatten(), gen.flatten()])
    lo = np.percentile(all_flat, 0.1)
    hi = np.percentile(all_flat, 99.9)
    half = max(abs(lo), abs(hi))
    return np.linspace(-half * 1.6, half * 1.6, n_bins)


def paired_quantiles(real, gen, n_q=300):
    probs  = np.linspace(0.5 / n_q, 1 - 0.5 / n_q, n_q)
    q_real = np.quantile(real.flatten(), probs)
    q_gen  = np.quantile(gen.flatten(),  probs)
    return q_real, q_gen


# Plot: histogram + Q-Q
def save_hist_qq(real, gen, title_suffix, filename, fitted_normal=False):
    """
    1 x 2 figure: histogram (left) and Q-Q plot (right).
    If fitted_normal=True, overlay a fitted normal curve on the histogram.
    """
    bins = _shared_bins(real, gen)
    real_f = real.flatten()
    gen_f  = gen.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Real vs Expectile-GAN  ({title_suffix})', fontsize=13, y=1.01)

    # histogram
    ax = axes[0]
    ax.hist(real_f, bins=bins, density=True, alpha=0.55,
            color=C['real'], label='Real data')
    ax.hist(gen_f,  bins=bins, density=True, alpha=0.55,
            color=C['expectile'], label=f'Expectile-GAN  ($\\tau={TAU}$)')
    if fitted_normal:
        mu_r, sig_r = real_f.mean(), real_f.std()
        x_pdf = np.linspace(bins[0], bins[-1], 400)
        ax.plot(x_pdf, stats.norm.pdf(x_pdf, mu_r, sig_r),
                'k--', linewidth=1.2, alpha=0.65,
                label=f'Fitted $\\mathcal{{N}}({mu_r:.4f},\\ {sig_r:.4f}^2)$')
    ax.set_xlabel('Daily log-return')
    ax.set_ylabel('Density')
    ax.set_title('Distribution')
    ax.legend()

    # Q-Q
    ax = axes[1]
    qr, qg = paired_quantiles(real, gen)
    lo = min(qr.min(), qg.min()) * 1.10
    hi = max(qr.max(), qg.max()) * 1.10
    ax.scatter(qr, qg, s=6, alpha=0.55, color=C['expectile'])
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.1, label='$y = x$')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Real quantiles')
    ax.set_ylabel('Expectile-GAN quantiles')
    ax.set_title('Q-Q Plot')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# Plot: left-tail comparison (1 x 2)
def save_tail_comparison(real, gen, filename, alpha=0.05, worst_pct=15, title=''):
    real_f = real.flatten()
    gen_f  = gen.flatten()

    tail_min = float(np.percentile(real_f, 0.05))
    tail_max = float(np.percentile(real_f, worst_pct))
    bins = np.linspace(tail_min, tail_max, 45)

    datasets = [
        (real, 'Real data',              C['real']),
        (gen,  f'Expectile-GAN ($\\tau={TAU}$)', C['expectile']),
    ]

    ymax = 0.0
    for d, _, _ in datasets:
        flat = d.flatten()
        thr  = np.percentile(flat, worst_pct)
        h, _ = np.histogram(flat[flat <= thr], bins=bins, density=True)
        if h.size:
            ymax = max(ymax, h.max())
    ymax *= 1.12

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'Left-tail comparison (worst {worst_pct}%  |  VaR and ES at $\\alpha={int(alpha*100)}\\%$)'
        + (f'  --  {title}' if title else ''),
        fontsize=12,
    )

    for ax, (data_arr, name, color) in zip(axes, datasets):
        flat = data_arr.flatten()
        thr  = np.percentile(flat, worst_pct)
        tail = flat[flat <= thr]
        var  = calculate_var(data_arr, alpha)
        es   = calculate_es(data_arr, alpha)

        ax.hist(tail, bins=bins, density=True, alpha=0.72, color=color, label=name)
        ax.axvline(var, color='crimson',    linestyle='--', linewidth=1.8,
                   label=f'VaR({int(alpha*100)}%) = {var:.4f}')
        ax.axvline(es,  color='darkorange', linestyle='--', linewidth=1.8,
                   label=f'ES({int(alpha*100)}%)  = {es:.4f}')
        ax.set_xlim(tail_min, tail_max)
        ax.set_ylim(0, ymax)
        ax.set_title(name)
        ax.set_xlabel('Daily log-return')
        ax.legend(fontsize=7.5)

    axes[0].set_ylabel('Density')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# Plot: sample paths (2-panel)
def save_sample_paths(real, gen, filename, n_paths=5, title=''):
    datasets = [
        (real, 'Real data',                       C['real']),
        (gen,  f'Expectile-GAN ($\\tau={TAU}$)',  C['expectile']),
    ]
    ylim = 0.03
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(
        f'{n_paths} random 50-day return windows' + (f'  --  {title}' if title else ''),
        fontsize=13,
    )
    rng = np.random.default_rng(seed=8)
    x   = np.arange(WINDOW_SIZE)

    for ax, (data_arr, name, color) in zip(axes, datasets):
        idx = rng.choice(data_arr.shape[0], n_paths, replace=False)
        for i in idx:
            ax.plot(x, data_arr[i], color=color, alpha=0.75, linewidth=0.9)
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_title(name, fontsize=11)
        ax.set_ylabel('Return', fontsize=9)
        ax.set_ylim(-ylim, ylim)

    axes[-1].set_xlabel('Time step (trading days within window)', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def worst_windows(data, n=5):
    idx = np.argsort(data.min(axis=1))[:n]
    return data[idx]


def save_crash_paths(real, gen, filename, n_paths=5, title=''):
    # Calculate ylim based only on the generated data
    ylim = float(np.abs(gen).max()) * 1.10

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f'{n_paths} worst-case 50-day windows (most negative single-day return)'
        + (f'  --  {title}' if title else ''),
        fontsize=12,
    )
    x = np.arange(WINDOW_SIZE)

    name = f'Expectile-GAN ($\\tau={TAU}$)'
    color = C['expectile']

    crash = worst_windows(gen, n=n_paths)
    for i in range(n_paths):
        ax.plot(x, crash[i], color=color, alpha=0.82, linewidth=0.9)

    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_title(name, fontsize=11)
    ax.set_ylabel('Return', fontsize=9)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel('Time step (trading days within window)', fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# numerical stats + LaTeX table
def _stats(data, tau=None, alphas=(0.01, 0.05)):
    if tau is None:
        tau = TAU
    flat = data.flatten()
    d = {
        'mean':     float(flat.mean()),
        'std':      float(flat.std()),
        'skewness': float(stats.skew(flat)),
        'kurtosis': float(stats.kurtosis(flat)),
        'expectile_tau': compute_expectile(data, tau),
    }
    for a in alphas:
        d[f'var_{int(a*100)}'] = calculate_var(data, a)
        d[f'es_{int(a*100)}']  = calculate_es(data,  a)
    return d


def print_and_save_stats(real, gen, dataset_name, tex_path, alphas=(0.01, 0.05)):
    sr = _stats(real, alphas=alphas)
    sg = _stats(gen,  alphas=alphas)

    hdr = f"{'Metric':<30} {'Real data':>15} {'Expectile-GAN':>15} {'Rel. error (%)':>16}"
    sep = '-' * len(hdr)
    print(f'\n{"="*len(hdr)}')
    print(f'Numerical summary  --  {dataset_name}  (tau = {TAU})')
    print('='*len(hdr))
    print(hdr)
    print(sep)

    def prow(label, key):
        rv = sr[key]; gv = sg[key]
        err = abs(gv - rv) / max(abs(rv), 1e-12) * 100
        print(f'{label:<30} {rv:>15.6f} {gv:>15.6f} {err:>15.2f}%')

    prow('Mean',           'mean')
    prow('Std Dev',        'std')
    prow('Skewness',       'skewness')
    prow('Kurtosis',       'kurtosis')
    prow(f'Expectile (tau={TAU})', 'expectile_tau')
    for a in alphas:
        prow(f'VaR ({int(a*100)}%)', f'var_{int(a*100)}')
        prow(f'ES  ({int(a*100)}%)', f'es_{int(a*100)}')
    print('='*len(hdr))

    # LaTeX table
    def fmt(v):
        return f'{v:.6f}'

    def pct(key):
        rv = sr[key]; gv = sg[key]
        return abs(gv - rv) / max(abs(rv), 1e-12) * 100

    lines = [
        r'\begin{table}[H]',
        r'  \centering',
        r'  \small',
        r'  \begin{tabular}{@{}lrrl@{}}',
        r'    \toprule',
        r'    \textbf{Metric} & \textbf{Real data} & \textbf{Expectile-GAN} & \textbf{Rel.\ error} \\',
        r'    \midrule',
    ]

    def trow(label, key):
        rv = sr[key]; gv = sg[key]
        err = pct(key)
        lines.append(f'    {label} & ${fmt(rv)}$ & ${fmt(gv)}$ & ${err:.2f}\\%$ \\\\')

    trow('Mean',                  'mean')
    trow('Std Dev',               'std')
    trow(f'Expectile ($\\tau={TAU}$)', 'expectile_tau')
    for a in alphas:
        trow(f'VaR ({int(a*100)}\\%)',  f'var_{int(a*100)}')
        trow(f'ES  ({int(a*100)}\\%)',  f'es_{int(a*100)}')
    tau_tag = f'tau{int(round(TAU*100)):02d}'
    name_lc = dataset_name.lower()
    if 'sp 500' in name_lc.replace('\\&', ''):
        base_lbl = 'sp500'
    elif 'gaussian' in name_lc:
        base_lbl = 'gaussian'
    else:
        base_lbl = (name_lc.replace(' ', '_').replace('&', '')
                          .replace('/', '').replace('\\', ''))
    label   = f'tab:expectile_gan_{base_lbl}' if abs(TAU - 0.05) < 1e-9 else f'tab:expectile_gan_{base_lbl}_{tau_tag}'
    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        f'  \\caption{{Expectile-GAN performance on {dataset_name} data ($\\tau={TAU}$, 500 training epochs).}}',
        f'  \\label{{{label}}}',
        r'\end{table}',
    ]
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  LaTeX table saved to {tex_path}')
    return sr, sg


#  main
def main(n_epochs=500, tau=DEFAULT_TAU, output_dir=None):
    global TAU, OUTPUT_DIR
    TAU = tau
    OUTPUT_DIR = output_dir if output_dir is not None else f'expectile_gan_tau{int(round(tau*100)):02d}_visuals' if tau != DEFAULT_TAU else DEFAULT_OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=' * 70)
    print('  expectile_gan_visuals.py')
    print(f'  Output: {OUTPUT_DIR}/   |   tau = {TAU}   |   epochs = {n_epochs}')
    print('=' * 70)

    # 1. Gaussian study
    print('\n-- Gaussian study -----------------------------------------')
    gauss_data = generate_gaussian_data(n_samples=1000, n_timesteps=50, std=0.02)
    print(f'  Gaussian data: shape={gauss_data.shape}  std={gauss_data.std():.5f}')

    print(f'  Training Expectile-GAN (tau={TAU}, {n_epochs} epochs) ...')
    g_gen, _, g_hist = train_expectile_gan(
        gauss_data, n_epochs=n_epochs, batch_size=64, latent_dim=100,
        lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
        tau_levels=[TAU], verbose=True,
    )
    g_samples = g_gen.generate(N_GEN).astype(np.float32)

    save_hist_qq(
        gauss_data, g_samples,
        title_suffix='Gaussian $\\mathcal{N}(0,\\ 0.02^2)$',
        filename='expectile_gaussian',
        fitted_normal=False,
    )
    save_tail_comparison(
        gauss_data, g_samples,
        filename='expectile_gaussian_tail',
        alpha=0.05, worst_pct=15,
        title='Gaussian',
    )
    gauss_tex = os.path.join(OUTPUT_DIR, 'expectile_gaussian_stats.tex')
    sr_g, sg_g = print_and_save_stats(gauss_data, g_samples, 'Gaussian', gauss_tex)

    # 2. S&P 500 study
    print('\n-- S&P 500 study --------------------------------------')
    raw_returns = load_sp500(CACHE_FILE)
    sp500_data  = create_rolling_windows(raw_returns, T=WINDOW_SIZE)
    print(f'  S&P 500 data: shape={sp500_data.shape}')

    print(f'  Training Expectile-GAN (tau={TAU}, {n_epochs} epochs) ...')
    sp_gen, _, sp_hist = train_expectile_gan(
        sp500_data, n_epochs=n_epochs, batch_size=64, latent_dim=100,
        lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
        tau_levels=[TAU], verbose=True,
    )
    sp_samples = sp_gen.generate(N_GEN).astype(np.float32)

    save_hist_qq(
        sp500_data, sp_samples,
        title_suffix='S\\&P 500 daily log-returns',
        filename='expectile_sp500',
    )
    save_tail_comparison(
        sp500_data, sp_samples,
        filename='expectile_sp500_tail',
        alpha=0.05, worst_pct=15,
        title='S\\&P 500',
    )
    save_sample_paths(
        sp500_data, sp_samples,
        filename='expectile_sp500_sample_paths',
        n_paths=5, title='S\\&P 500',
    )
    save_crash_paths(sp500_data,
         sp_samples,
        filename='expectile_sp500_crash_paths',
        n_paths=3, title='S\\&P 500',
    )
    sp500_tex = os.path.join(OUTPUT_DIR, 'expectile_sp500_stats.tex')
    sr_sp, sg_sp = print_and_save_stats(sp500_data, sp_samples, 'S\\&P 500', sp500_tex)

    # summary
    print('\n' + '=' * 70)
    print('  Done.  Files in  ' + OUTPUT_DIR + '/')
    print('  Figures:')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f'    {f}')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Expectile-GAN visuals for Chapter 7 of the thesis.'
    )
    parser.add_argument('--epochs', type=int, default=500,
                        help='Training epochs (default 500; use 50 for a quick test)')
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU,
                        help='Expectile level tau (default 0.05; common alternatives 0.01, 0.10)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: expectile_gan_visuals for tau=0.05, '
                             'expectile_gan_tau<NN>_visuals otherwise)')
    args = parser.parse_args()
    main(n_epochs=args.epochs, tau=args.tau, output_dir=args.output_dir)
