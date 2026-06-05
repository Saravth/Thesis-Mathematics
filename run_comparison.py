"""
Runner script to train both WGAN-GP and Tail-GAN and compare results.

This script:
1. Generates synthetic Gaussian returns data
2. Trains WGAN-GP (Algorithm 3)
3. Trains Tail-GAN (Algorithm 4), using returns
4. Compares tail risk metrics (VaR, ES)
5. Creates visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import our implementations
from wgan_gp import train_wgan_gp, generate_gaussian_data as gen_data_wgan
from tail_gan import train_tailgan, generate_gaussian_data as gen_data_tail, calculate_var, calculate_es


def create_output_dir():
    """Create output directory for results."""
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_data(n_samples=1000, n_timesteps=50, std=0.02):
    """Generate synthetic Gaussian returns data."""
    np.random.seed(42)  # For reproducibility
    return np.random.normal(0, std, (n_samples, n_timesteps)).astype(np.float32)


def plot_training_curves(wgan_history, tailgan_history, output_dir):
    """Plot training loss curves for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # WGAN-GP losses
    ax1 = axes[0, 0]
    ax1.plot(wgan_history['critic_loss'], label='Critic Loss', alpha=0.8)
    ax1.plot(wgan_history['generator_loss'], label='Generator Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('WGAN-GP Training Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # WGAN-GP Wasserstein distance
    ax2 = axes[0, 1]
    ax2.plot(wgan_history['wasserstein_dist'], label='Wasserstein Distance', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Distance')
    ax2.set_title('WGAN-GP Wasserstein Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Tail-GAN losses
    ax3 = axes[1, 0]
    ax3.plot(tailgan_history['d_loss'], label='Discriminator Loss', alpha=0.8)
    ax3.plot(tailgan_history['g_loss'], label='Generator Loss', alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Tail-GAN Training Losses')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Tail-GAN VaR/ES estimates
    ax4 = axes[1, 1]
    ax4.plot(tailgan_history['var_estimate'], label='VaR Estimate', alpha=0.8)
    ax4.plot(tailgan_history['es_estimate'], label='ES Estimate', alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Value')
    ax4.set_title('Tail-GAN VaR/ES Estimates During Training')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Saved training curves to {output_dir}/training_curves.png")


def plot_distribution_comparison(real_data, wgan_data, tailgan_data, output_dir):
    """Plot distribution comparisons."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Flatten all data
    real_flat = real_data.flatten()
    wgan_flat = wgan_data.flatten()
    tailgan_flat = tailgan_data.flatten()

    # Histograms
    bins = np.linspace(real_flat.min(), real_flat.max(), 50)

    ax1 = axes[0, 0]
    ax1.hist(real_flat, bins=bins, alpha=0.7, label='Real', density=True)
    ax1.hist(wgan_flat, bins=bins, alpha=0.5, label='WGAN-GP', density=True)
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Density')
    ax1.set_title('Real vs WGAN-GP Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.hist(real_flat, bins=bins, alpha=0.7, label='Real', density=True)
    ax2.hist(tailgan_flat, bins=bins, alpha=0.5, label='Tail-GAN', density=True)
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Density')
    ax2.set_title('Real vs Tail-GAN Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    ax3.hist(wgan_flat, bins=bins, alpha=0.7, label='WGAN-GP', density=True)
    ax3.hist(tailgan_flat, bins=bins, alpha=0.5, label='Tail-GAN', density=True)
    ax3.set_xlabel('Return')
    ax3.set_ylabel('Density')
    ax3.set_title('WGAN-GP vs Tail-GAN Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Q-Q plots
    from scipy import stats

    ax4 = axes[1, 0]
    sorted_real = np.sort(real_flat)
    sorted_wgan = np.sort(wgan_flat)[:len(sorted_real)] if len(wgan_flat) > len(sorted_real) else np.sort(wgan_flat)
    if len(sorted_wgan) < len(sorted_real):
        sorted_real = sorted_real[:len(sorted_wgan)]
    ax4.scatter(sorted_real, sorted_wgan, alpha=0.3, s=1)
    lims = [min(sorted_real.min(), sorted_wgan.min()), max(sorted_real.max(), sorted_wgan.max())]
    ax4.plot(lims, lims, 'r--', label='y=x')
    ax4.set_xlabel('Real Quantiles')
    ax4.set_ylabel('WGAN-GP Quantiles')
    ax4.set_title('Q-Q Plot: Real vs WGAN-GP')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    sorted_tailgan = np.sort(tailgan_flat)[:len(sorted_real)] if len(tailgan_flat) > len(sorted_real) else np.sort(tailgan_flat)
    if len(sorted_tailgan) < len(sorted_real):
        sorted_real_t = sorted_real[:len(sorted_tailgan)]
    else:
        sorted_real_t = sorted_real
    ax5.scatter(sorted_real_t, sorted_tailgan, alpha=0.3, s=1)
    lims = [min(sorted_real_t.min(), sorted_tailgan.min()), max(sorted_real_t.max(), sorted_tailgan.max())]
    ax5.plot(lims, lims, 'r--', label='y=x')
    ax5.set_xlabel('Real Quantiles')
    ax5.set_ylabel('Tail-GAN Quantiles')
    ax5.set_title('Q-Q Plot: Real vs Tail-GAN')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Statistics comparison
    ax6 = axes[1, 2]
    stats_names = ['Mean', 'Std', 'Skewness', 'Kurtosis']
    real_stats = [real_flat.mean(), real_flat.std(),
                  stats.skew(real_flat), stats.kurtosis(real_flat)]
    wgan_stats = [wgan_flat.mean(), wgan_flat.std(),
                  stats.skew(wgan_flat), stats.kurtosis(wgan_flat)]
    tailgan_stats = [tailgan_flat.mean(), tailgan_flat.std(),
                     stats.skew(tailgan_flat), stats.kurtosis(tailgan_flat)]

    x = np.arange(len(stats_names))
    width = 0.25
    ax6.bar(x - width, real_stats, width, label='Real', alpha=0.8)
    ax6.bar(x, wgan_stats, width, label='WGAN-GP', alpha=0.8)
    ax6.bar(x + width, tailgan_stats, width, label='Tail-GAN', alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(stats_names)
    ax6.set_title('Statistical Moments Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved distribution comparison to {output_dir}/distribution_comparison.png")


def plot_tail_comparison(real_data, wgan_data, tailgan_data, alpha, output_dir):
    """Plot tail risk comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    real_flat = real_data.flatten()
    wgan_flat = wgan_data.flatten()
    tailgan_flat = tailgan_data.flatten()

    # Calculate VaR and ES
    real_var = calculate_var(real_data, alpha)
    real_es = calculate_es(real_data, alpha)
    wgan_var = calculate_var(wgan_data, alpha)
    wgan_es = calculate_es(wgan_data, alpha)
    tailgan_var = calculate_var(tailgan_data, alpha)
    tailgan_es = calculate_es(tailgan_data, alpha)

    # Left tail histograms with VaR/ES lines
    tail_cutoff = np.percentile(real_flat, 15)  # Show left 15%
    bins = np.linspace(real_flat.min(), tail_cutoff, 40)

    for ax, data_flat, var, es, title in [
        (axes[0], real_flat, real_var, real_es, 'Real Data'),
        (axes[1], wgan_flat, wgan_var, wgan_es, 'WGAN-GP'),
        (axes[2], tailgan_flat, tailgan_var, tailgan_es, 'Tail-GAN')
    ]:
        ax.hist(data_flat[data_flat <= tail_cutoff], bins=bins, density=True, alpha=0.7)
        ax.axvline(var, color='r', linestyle='--', linewidth=2,
                   label=f'VaR ({alpha*100:.0f}%): {var:.4f}')
        ax.axvline(es, color='orange', linestyle='--', linewidth=2,
                   label=f'ES ({alpha*100:.0f}%): {es:.4f}')
        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        ax.set_title(f'{title} - Left Tail')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tail_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved tail comparison to {output_dir}/tail_comparison.png")

    return {
        'real_var': real_var, 'real_es': real_es,
        'wgan_var': wgan_var, 'wgan_es': wgan_es,
        'tailgan_var': tailgan_var, 'tailgan_es': tailgan_es
    }


def plot_sample_paths(real_data, wgan_data, tailgan_data, output_dir, n_paths=5):
    """Plot sample return paths."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for ax, data, title in [
        (axes[0], real_data, 'Real Return Paths'),
        (axes[1], wgan_data, 'WGAN-GP Generated Paths'),
        (axes[2], tailgan_data, 'Tail-GAN Generated Paths')
    ]:
        for i in range(min(n_paths, len(data))):
            ax.plot(data[i], alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Return')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_paths.png'), dpi=150)
    plt.close()
    print(f"Saved sample paths to {output_dir}/sample_paths.png")


def main():
    print("=" * 70)
    print("IMPROVED GAN IMPLEMENTATIONS - COMPARISON STUDY")
    print("Following Algorithms 2 (WGAN-GP) and 3 (Tail-GAN) from thesis")
    print("=" * 70)

    output_dir = create_output_dir()
    alpha = 0.05  # 5% VaR/ES

    # =========================================================================
    # Generate Data
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 1: Generating synthetic Gaussian returns data")
    print("-" * 70)

    data = generate_data(n_samples=1000, n_timesteps=50, std=0.02)
    print(f"Data shape: {data.shape}")
    print(f"Data statistics:")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Std:  {data.std():.6f}")
    print(f"  VaR ({alpha*100:.0f}%): {calculate_var(data, alpha):.6f}")
    print(f"  ES ({alpha*100:.0f}%):  {calculate_es(data, alpha):.6f}")

    # =========================================================================
    # Train WGAN-GP
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Training WGAN-GP (Algorithm 2 from thesis)")
    print("-" * 70)
    print("Parameters: lr=0.0001, n_critic=5, lambda_gp=10, beta1=0, beta2=0.9")

    wgan_generator, wgan_history = train_wgan_gp(
        data,
        n_epochs=500,  # More epochs for better convergence
        batch_size=64,
        latent_dim=100,
        lr=1e-4,
        n_critic=5,
        lambda_gp=10,
        beta1=0.0,
        beta2=0.9,
        verbose=True
    )

    # Generate samples
    wgan_samples = wgan_generator.generate(1000)
    print(f"\nWGAN-GP generated {wgan_samples.shape[0]} samples")

    # =========================================================================
    # Train Tail-GAN
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 3: Training Tail-GAN (Algorithm 3 from thesis, using returns)")
    print("-" * 70)
    print(f"Parameters: alphas={[alpha]}, using Fissler-Ziegel scoring function")

    tailgan_generator, tailgan_discriminator, tailgan_history = train_tailgan(
        data,
        n_epochs=300,  # More epochs for better convergence
        batch_size=64,
        latent_dim=100,
        lr_d=1e-5,
        lr_g=1e-5,
        lambda_dual=1.0,
        alphas=[alpha],
        verbose=True
    )

    # Generate samples
    tailgan_samples = tailgan_generator.generate(1000)
    print(f"\nTail-GAN generated {tailgan_samples.shape[0]} samples")

    # =========================================================================
    # Create Visualizations
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 4: Creating visualizations")
    print("-" * 70)

    plot_training_curves(wgan_history, tailgan_history, output_dir)
    plot_distribution_comparison(data, wgan_samples, tailgan_samples, output_dir)
    tail_metrics = plot_tail_comparison(data, wgan_samples, tailgan_samples, alpha, output_dir)
    plot_sample_paths(data, wgan_samples, tailgan_samples, output_dir)

    # =========================================================================
    # Final Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON - TAIL RISK METRICS")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Real':>12} {'WGAN-GP':>12} {'Tail-GAN':>12}")
    print("-" * 56)
    print(f"{'Mean':.<20} {data.mean():>12.6f} {wgan_samples.mean():>12.6f} {tailgan_samples.mean():>12.6f}")
    print(f"{'Std':.<20} {data.std():>12.6f} {wgan_samples.std():>12.6f} {tailgan_samples.std():>12.6f}")
    print(f"{'VaR (5%)':.<20} {tail_metrics['real_var']:>12.6f} {tail_metrics['wgan_var']:>12.6f} {tail_metrics['tailgan_var']:>12.6f}")
    print(f"{'ES (5%)':.<20} {tail_metrics['real_es']:>12.6f} {tail_metrics['wgan_es']:>12.6f} {tail_metrics['tailgan_es']:>12.6f}")

    # Calculate errors
    print("\n" + "-" * 56)
    print("ERRORS (relative to Real)")
    print("-" * 56)

    wgan_var_err = abs(tail_metrics['wgan_var'] - tail_metrics['real_var']) / abs(tail_metrics['real_var']) * 100
    wgan_es_err = abs(tail_metrics['wgan_es'] - tail_metrics['real_es']) / abs(tail_metrics['real_es']) * 100
    tailgan_var_err = abs(tail_metrics['tailgan_var'] - tail_metrics['real_var']) / abs(tail_metrics['real_var']) * 100
    tailgan_es_err = abs(tail_metrics['tailgan_es'] - tail_metrics['real_es']) / abs(tail_metrics['real_es']) * 100

    print(f"{'VaR Error':.<20} {'-':>12} {wgan_var_err:>11.2f}% {tailgan_var_err:>11.2f}%")
    print(f"{'ES Error':.<20} {'-':>12} {wgan_es_err:>11.2f}% {tailgan_es_err:>11.2f}%")

    print("\n" + "=" * 70)
    print(f"Results saved to: {os.path.abspath(output_dir)}/")
    print("=" * 70)

    return wgan_generator, tailgan_generator, {
        'data': data,
        'wgan_samples': wgan_samples,
        'tailgan_samples': tailgan_samples,
        'tail_metrics': tail_metrics
    }


if __name__ == "__main__":
    wgan_gen, tailgan_gen, results = main()
