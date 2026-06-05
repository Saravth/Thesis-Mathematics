"""
Trains a multi-tau Expectile-GAN with tau_levels = [0.01, 0.05, 0.10] on:
  1. Synthetic Gaussian data:  N(0, 0.02^2), 1000 samples x 50 timesteps
  2. S&P 500 daily log-returns: rolling 50-day windows (2016-2023)

Produces figures and a results table comparing expectiles, VaRs and ESs across
all three tau levels. Matches to expectile_gan_visuals.py  (which only handles a single tau).

Output folder: expectile_gan_taus_visuals/

Figures produced:
  expectile_gaussian_taus.png          hist + Q-Q (multi-tau)
  expectile_gaussian_taus_tail.png     tail comparison at alpha in {0.01, 0.05, 0.10}
  expectile_gaussian_taus_stats.tex    LaTeX table with all three tau levels
  expectile_sp500_taus.png             hist + Q-Q (multi-tau)
  expectile_sp500_taus_tail.png        tail comparison
  expectile_sp500_taus_sample_paths.png
  expectile_sp500_taus_crash_paths.png
  expectile_sp500_taus_stats.tex
  expectile_gaussian_taus_bars.png     side-by-side bar comparison of expectiles
  expectile_sp500_taus_bars.png        idem on S&P 500

"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from expectile_gan import train_expectile_gan, compute_expectile, calculate_var, calculate_es

# constants
OUTPUT_DIR   = 'expectile_gan_taus_visuals'
CACHE_FILE   = 'sp500_cache.npy'
WINDOW_SIZE  = 50
N_GEN        = 2000
TAU_LEVELS   = [0.01, 0.05, 0.10]
ALPHA_LEVELS = [0.01, 0.05, 0.10]

C = {
    'real': '#1f77b4',
    'gen':  '#2ca02c',          # green for multi-tau, distinguishing from single-tau red
    'gen_alt': '#9467bd',
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


# shared bins
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


# Plot: histogram + Q-Q (multi-tau)
def save_hist_qq(real, gen, title_suffix, filename):
    bins = _shared_bins(real, gen)
    real_f = real.flatten()
    gen_f  = gen.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'Real vs multi-$\\tau$ Expectile-GAN  ($\\tau \\in \\{{0.01, 0.05, 0.10\\}}$, '
        f'{title_suffix})',
        fontsize=12, y=1.01)

    ax = axes[0]
    ax.hist(real_f, bins=bins, density=True, alpha=0.55,
            color=C['real'], label='Real data')
    ax.hist(gen_f,  bins=bins, density=True, alpha=0.55,
            color=C['gen'],  label='Multi-$\\tau$ Expectile-GAN')
    ax.set_xlabel('Daily log-return')
    ax.set_ylabel('Density')
    ax.set_title('Distribution')
    ax.legend()

    ax = axes[1]
    qr, qg = paired_quantiles(real, gen)
    lo = min(qr.min(), qg.min()) * 1.10
    hi = max(qr.max(), qg.max()) * 1.10
    ax.scatter(qr, qg, s=6, alpha=0.55, color=C['gen'])
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.1, label='$y = x$')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Real quantiles')
    ax.set_ylabel('Generated quantiles')
    ax.set_title('Q-Q Plot')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# Plot: tail comparison across multiple alpha levels
def save_tail_comparison_multi(real, gen, filename, alphas=ALPHA_LEVELS,
                               worst_pct=15, title=''):
    """
    One row, len(alphas) columns.  Each panel zooms on the worst worst_pct of
    real returns and overlays VaR and ES at one alpha level for both real and
    generated data.  Both histograms share each panel.
    """
    real_f = real.flatten()
    gen_f  = gen.flatten()
    tail_min = float(np.percentile(real_f, 0.05))
    tail_max = float(np.percentile(real_f, worst_pct))
    bins = np.linspace(tail_min, tail_max, 45)

    fig, axes = plt.subplots(1, len(alphas), figsize=(5.0 * len(alphas), 5))
    fig.suptitle(
        f'Multi-$\\tau$ Expectile-GAN  --  tail comparison at $\\alpha \\in '
        f'\\{{{", ".join(f"{int(a*100)}\\%" for a in alphas)}\\}}$'
        + (f'  --  {title}' if title else ''),
        fontsize=12, y=1.02)

    for ax, alpha in zip(axes, alphas):
        real_tail = real_f[real_f <= np.percentile(real_f, worst_pct)]
        gen_tail  = gen_f[gen_f  <= np.percentile(real_f, worst_pct)]

        ax.hist(real_tail, bins=bins, density=True, alpha=0.55,
                color=C['real'], label='Real data')
        ax.hist(gen_tail,  bins=bins, density=True, alpha=0.55,
                color=C['gen'],  label='Multi-$\\tau$ E-GAN')

        var_r = calculate_var(real, alpha); es_r = calculate_es(real, alpha)
        var_g = calculate_var(gen,  alpha); es_g = calculate_es(gen,  alpha)

        ax.axvline(var_r, color=C['real'], linestyle='--', linewidth=1.4,
                   label=f'Real VaR({int(alpha*100)}%) = {var_r:.4f}')
        ax.axvline(es_r,  color=C['real'], linestyle=':',  linewidth=1.4,
                   label=f'Real ES({int(alpha*100)}%)  = {es_r:.4f}')
        ax.axvline(var_g, color=C['gen'],  linestyle='--', linewidth=1.4,
                   label=f'E-GAN VaR = {var_g:.4f}')
        ax.axvline(es_g,  color=C['gen'],  linestyle=':',  linewidth=1.4,
                   label=f'E-GAN ES  = {es_g:.4f}')

        ax.set_xlim(tail_min, tail_max)
        ax.set_title(f'$\\alpha = {alpha}$')
        ax.set_xlabel('Daily log-return')
        ax.legend(fontsize=7)

    axes[0].set_ylabel('Density')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# Plot: sample paths and crash paths
def save_sample_paths(real, gen, filename, n_paths=5, title=''):
    datasets = [
        (real, 'Real data',                C['real']),
        (gen,  'Multi-$\\tau$ E-GAN',      C['gen']),
    ]
    ylim = 0.03 # good enough to look at
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(
        f'{n_paths} random 50-day return windows' + (f'  --  {title}' if title else ''),
        fontsize=13)
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


def save_crash_paths(gen, filename, n_paths=3, title=''):
    ylim = float(np.abs(gen).max()) * 1.10
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f'{n_paths} worst-case 50-day windows (most negative single-day return)'
        + (f'  --  {title}' if title else ''),
        fontsize=12)
    x = np.arange(WINDOW_SIZE)
    crash = worst_windows(gen, n=n_paths)
    for i in range(n_paths):
        ax.plot(x, crash[i], color=C['gen'], alpha=0.82, linewidth=0.9)
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_title('Multi-$\\tau$ Expectile-GAN', fontsize=11)
    ax.set_ylabel('Return', fontsize=9)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel('Time step (trading days within window)', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


#  Plot: bar plot of expectiles at all taus
def save_expectile_bars(real, gen, filename, taus=TAU_LEVELS, title=''):
    """
    Side-by-side bars: expectile estimate at each tau, real vs generated.
    """
    real_e = [compute_expectile(real, t) for t in taus]
    gen_e  = [compute_expectile(gen,  t) for t in taus]

    x = np.arange(len(taus))
    width = 0.35 # looks good

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, real_e, width, color=C['real'], alpha=0.85, label='Real data')
    ax.bar(x + width / 2, gen_e,  width, color=C['gen'],  alpha=0.85, label='Multi-$\\tau$ E-GAN')

    for xi, rv in zip(x - width / 2, real_e):
        ax.text(xi, rv * 1.02 if rv < 0 else rv + 0.0005, f'{rv:.4f}',
                ha='center', va='top' if rv < 0 else 'bottom', fontsize=8)
    for xi, gv in zip(x + width / 2, gen_e):
        ax.text(xi, gv * 1.02 if gv < 0 else gv + 0.0005, f'{gv:.4f}',
                ha='center', va='top' if gv < 0 else 'bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'$\\tau = {t}$' for t in taus])
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_ylabel('Expectile $\\mu_\\tau$')
    ax.set_title(f'Expectile estimates at $\\tau \\in \\{{0.01, 0.05, 0.10\\}}$'
                 + (f'  --  {title}' if title else ''),
                 fontsize=11)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# Stats + LaTeX table for multi-tau
def _stats_multi(data, taus=TAU_LEVELS, alphas=ALPHA_LEVELS):
    flat = data.flatten()
    d = {
        'mean':     float(flat.mean()),
        'std':      float(flat.std()),
        'skewness': float(stats.skew(flat)),
        'kurtosis': float(stats.kurtosis(flat)),
    }
    for t in taus:
        d[f'expectile_{int(t*100):02d}'] = compute_expectile(data, t)
    for a in alphas:
        d[f'var_{int(a*100):02d}'] = calculate_var(data, a)
        d[f'es_{int(a*100):02d}']  = calculate_es(data,  a)
    return d


def print_and_save_stats_multi(real, gen, dataset_name, tex_path,
                               taus=TAU_LEVELS, alphas=ALPHA_LEVELS):
    sr = _stats_multi(real, taus=taus, alphas=alphas)
    sg = _stats_multi(gen,  taus=taus, alphas=alphas)

    hdr = f"{'Metric':<28} {'Real data':>15} {'Multi-tau E-GAN':>18} {'Rel. error (%)':>16}"
    sep = '-' * len(hdr)
    print(f'\n{"="*len(hdr)}')
    print(f'Numerical summary  --  {dataset_name}  (multi-tau: {taus})')
    print('='*len(hdr))
    print(hdr); print(sep)

    def prow(label, key):
        rv = sr[key]; gv = sg[key]
        err = abs(gv - rv) / max(abs(rv), 1e-12) * 100
        print(f'{label:<28} {rv:>15.6f} {gv:>18.6f} {err:>15.2f}%')

    prow('Mean',     'mean')
    prow('Std Dev',  'std')
    prow('Skewness', 'skewness')
    prow('Kurtosis', 'kurtosis')
    for t in taus:
        prow(f'Expectile (tau={t})', f'expectile_{int(t*100):02d}')
    for a in alphas:
        prow(f'VaR ({int(a*100)}%)', f'var_{int(a*100):02d}')
        prow(f'ES  ({int(a*100)}%)', f'es_{int(a*100):02d}')
    print('='*len(hdr))

    # LaTeX table
    def fmt(v):  return f'{v:.6f}'
    def pct(key):
        rv = sr[key]; gv = sg[key]
        return abs(gv - rv) / max(abs(rv), 1e-12) * 100

    lines = [
        r'\begin{table}[H]',
        r'  \centering',
        r'  \small',
        r'  \begin{tabular}{@{}lrrl@{}}',
        r'    \toprule',
        r'    \textbf{Metric} & \textbf{Real data} & \textbf{Multi-$\tau$ E-GAN} & \textbf{Rel.\ error} \\',
        r'    \midrule',
    ]

    def trow(label, key):
        rv = sr[key]; gv = sg[key]
        lines.append(f'    {label} & ${fmt(rv)}$ & ${fmt(gv)}$ & ${pct(key):.2f}\\%$ \\\\')

    trow('Mean',    'mean')
    trow('Std Dev', 'std')
    for t in taus:
        trow(f'Expectile ($\\tau = {t}$)', f'expectile_{int(t*100):02d}')
    for a in alphas:
        trow(f'VaR ({int(a*100)}\\%)', f'var_{int(a*100):02d}')
        trow(f'ES  ({int(a*100)}\\%)', f'es_{int(a*100):02d}')

    name_lc = dataset_name.lower()
    if 'sp 500' in name_lc.replace('\\&', ''):
        base_lbl = 'sp500'
    elif 'gaussian' in name_lc:
        base_lbl = 'gaussian'
    else:
        base_lbl = (name_lc.replace(' ', '_').replace('&', '')
                          .replace('/', '').replace('\\', ''))
    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        f'  \\caption{{Multi-$\\tau$ Expectile-GAN performance on {dataset_name} data '
        f'($\\tau \\in \\{{0.01, 0.05, 0.10\\}}$, 500 training epochs).}}',
        f'  \\label{{tab:expectile_gan_taus_{base_lbl}}}',
        r'\end{table}',
    ]
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  LaTeX table saved to {tex_path}')
    return sr, sg


# main
def main(n_epochs=500):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=' * 70)
    print('  expectile_gan_taus_visuals.py')
    print(f'  Output: {OUTPUT_DIR}/   |   tau_levels = {TAU_LEVELS}   |   epochs = {n_epochs}')
    print('=' * 70)

    # 1. Gaussian study
    print('\n Gaussian study (multi-tau) ')
    gauss_data = generate_gaussian_data(n_samples=1000, n_timesteps=50, std=0.02)
    print(f'  Gaussian data: shape={gauss_data.shape}  std={gauss_data.std():.5f}')

    print(f'  Training multi-tau Expectile-GAN (tau_levels={TAU_LEVELS}, '
          f'{n_epochs} epochs) ...')
    g_gen, _, _ = train_expectile_gan(
        gauss_data, n_epochs=n_epochs, batch_size=64, latent_dim=100,
        lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
        tau_levels=TAU_LEVELS, verbose=True,
    )
    g_samples = g_gen.generate(N_GEN).astype(np.float32)

    save_hist_qq(gauss_data, g_samples,
                 title_suffix='Gaussian $\\mathcal{N}(0,\\ 0.02^2)$',
                 filename='expectile_gaussian_taus')
    save_tail_comparison_multi(gauss_data, g_samples,
                               filename='expectile_gaussian_taus_tail',
                               alphas=ALPHA_LEVELS, worst_pct=15, title='Gaussian')
    save_expectile_bars(gauss_data, g_samples,
                        filename='expectile_gaussian_taus_bars', title='Gaussian')
    gauss_tex = os.path.join(OUTPUT_DIR, 'expectile_gaussian_taus_stats.tex')
    print_and_save_stats_multi(gauss_data, g_samples, 'Gaussian', gauss_tex)

    # 2. S&P 500 study
    print('\n S&P 500 study (multi-tau)')
    raw_returns = load_sp500(CACHE_FILE)
    sp500_data  = create_rolling_windows(raw_returns, T=WINDOW_SIZE)
    print(f'  S&P 500 data: shape={sp500_data.shape}')

    print(f'  Training multi-tau Expectile-GAN (tau_levels={TAU_LEVELS}, '
          f'{n_epochs} epochs) ...')
    sp_gen, _, _ = train_expectile_gan(
        sp500_data, n_epochs=n_epochs, batch_size=64, latent_dim=100,
        lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
        tau_levels=TAU_LEVELS, verbose=True,
    )
    sp_samples = sp_gen.generate(N_GEN).astype(np.float32)

    save_hist_qq(sp500_data, sp_samples,
                 title_suffix='S\\&P 500 daily log-returns',
                 filename='expectile_sp500_taus')
    save_tail_comparison_multi(sp500_data, sp_samples,
                               filename='expectile_sp500_taus_tail',
                               alphas=ALPHA_LEVELS, worst_pct=15, title='S\\&P 500')
    save_sample_paths(sp500_data, sp_samples,
                      filename='expectile_sp500_taus_sample_paths',
                      n_paths=5, title='S\\&P 500')
    save_crash_paths(sp_samples,
                     filename='expectile_sp500_taus_crash_paths',
                     n_paths=3, title='S\\&P 500')
    save_expectile_bars(sp500_data, sp_samples,
                        filename='expectile_sp500_taus_bars', title='S\\&P 500')
    sp500_tex = os.path.join(OUTPUT_DIR, 'expectile_sp500_taus_stats.tex')
    print_and_save_stats_multi(sp500_data, sp_samples, 'S\\&P 500', sp500_tex)

    # summary
    print('\n' + '=' * 70)
    print('  Done.  Files in  ' + OUTPUT_DIR + '/')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f'    {f}')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-tau Expectile-GAN visuals '
                    '(tau_levels = [0.01, 0.05, 0.10]).'
    )
    parser.add_argument('--epochs', type=int, default=500,
                        help='Training epochs')
    args = parser.parse_args()
    main(n_epochs=args.epochs)
