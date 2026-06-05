"""
Expectile-GAN: An Alternative Tail-Sensitive Generative Adversarial Network

Uses the asymmetric squared loss (strictly consistent scoring function for
expectiles) instead of the Fissler-Ziegel score used in Tail-GAN.

References:
    - Newey & Powell (1987): Asymmetric Least Squares Estimation and Testing
    - Bellini & Di Bernardino (2017): Generalized quantiles as risk measures
    - Ziegel (2016): Coherence and elicitability
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, output_dim),
        )

    def forward(self, z):
        return self.model(z)

    def generate(self, n_samples, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            samples = self(z).cpu().numpy()
        self.train()
        return samples


# ---------------------------------------------------------------------------
# Expectile Discriminator
# ---------------------------------------------------------------------------
class ExpectileDiscriminator(nn.Module):
    """
    Outputs one expectile estimate per tau level.
    For L tau levels, the output is in \mathbb{R}^L.
    """

    def __init__(self, input_dim=50, tau_levels=[0.04]):
        super().__init__()
        self.tau_levels = tau_levels
        self.n_outputs = len(tau_levels)

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, self.n_outputs),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Asymmetric Squared Loss (expectile scoring function)
# ---------------------------------------------------------------------------
def asymmetric_squared_loss(m, y, tau):
    """
    S_tau(m, y) = tau * (y - m)_+^2 + (1 - tau) * (m - y)_+^2

    Strictly consistent scoring function for the tau-expectile.

    Args:
        m: expectile estimate (scalar or tensor)
        y: observations (tensor)
        tau: expectile level in (0, 1)
    Returns:
        mean score over observations
    """
    residual = y - m
    positive_part = torch.clamp(residual, min=0)
    negative_part = torch.clamp(-residual, min=0)
    score = tau * positive_part ** 2 + (1 - tau) * negative_part ** 2
    return score.mean()


class ExpectileScorer(nn.Module):
    """
    Multi-level expectile scoring function.

    S(m_1, ..., m_L, y) = sum_{l=1}^{L} S_{tau_l}(m_l, y)

    Each S_{tau_l} is individually strictly consistent for mu_{tau_l},
    so the combined score is strictly consistent for the vector of expectiles.
    """

    def __init__(self, tau_levels=[0.04]):
        super().__init__()
        self.tau_levels = tau_levels

    def forward(self, expectile_estimates, returns):
        """
        Args:
            expectile_estimates: (batch_size, L) tensor of expectile estimates
            returns: (batch_size, T) tensor of return series
        Returns:
            scalar combined score
        """
        total_score = 0.0
        y = returns.flatten()

        for i, tau in enumerate(self.tau_levels):
            m = expectile_estimates[:, i].mean()
            score = asymmetric_squared_loss(m, y, tau)
            total_score = total_score + score

        return total_score


# ---------------------------------------------------------------------------
# Compute empirical expectiles from data
# ---------------------------------------------------------------------------
def compute_expectile(data, tau, max_iter=200, tol=1e-8):
    """
    Compute the tau-expectile of a 1D array by iteratively solving
    the balance equation:
        tau * E[(X - m)^+] = (1 - tau) * E[(m - X)^+]

    Uses the iterative weighted least squares algorithm:
        m_{k+1} = sum_i w_i * x_i / sum_i w_i
    where w_i = tau if x_i > m_k, else (1 - tau).

    Args:
        data: 1D numpy array
        tau: expectile level in (0, 1)
        max_iter: maximum iterations
        tol: convergence tolerance
    Returns:
        the tau-expectile (float)
    """
    x = data.flatten().astype(np.float64)
    m = np.mean(x)  # start at the mean (the 0.5-expectile)

    for _ in range(max_iter):
        weights = np.where(x > m, tau, 1 - tau)
        m_new = np.sum(weights * x) / np.sum(weights)
        if abs(m_new - m) < tol:
            break
        m = m_new

    return float(m)


def compute_expectile_for_normal(sigma, tau, max_iter=2000, tol=1e-12):
    """
    Compute the tau-expectile of N(0, sigma^2) using a large sample
    from the standard normal and scaling.
    """
    np.random.seed(12345)
    z = np.random.normal(0, 1, 1_000_000)
    c_tau = compute_expectile(z, tau, max_iter=max_iter, tol=tol)
    return sigma * c_tau


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_expectile_gan(data, n_epochs=500, batch_size=64, latent_dim=100,
                        lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
                        tau_levels=[0.04], verbose=True):
    """
    Train an Expectile-GAN on return data.

    The discriminator learns the true expectiles of the data at multiple
    tau levels. The generator learns to produce returns whose expectiles
    match those of the real data.

    Args:
        data: numpy array of shape (n_samples, n_timesteps)
        n_epochs: number of training epochs
        batch_size: mini-batch size
        latent_dim: dimension of the latent noise vector z
        lr_d: learning rate for discriminator
        lr_g: learning rate for generator
        lambda_dual: weight for the real-data score in discriminator loss
        tau_levels: list of expectile levels to target
        verbose: print progress every 50 epochs

    Returns:
        (DenormalizedGenerator, discriminator, history)
    """
    device = get_device()
    n_timesteps = data.shape[1]

    # Normalize data (same pattern as tail_gan.py)
    data_mean = data.mean()
    data_std = data.std()
    data_normalized = (data - data_mean) / data_std

    # Models
    generator = Generator(latent_dim=latent_dim, output_dim=n_timesteps).to(device)
    discriminator = ExpectileDiscriminator(
        input_dim=n_timesteps, tau_levels=tau_levels
    ).to(device)
    scorer = ExpectileScorer(tau_levels=tau_levels).to(device)

    # Optimizers (same betas as Tail-GAN)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # DataLoader
    tensor_data = torch.FloatTensor(data_normalized)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # History
    history = {
        'd_loss': [],
        'g_loss': [],
        'expectile_estimates': {tau: [] for tau in tau_levels},
    }

    for epoch in range(n_epochs):
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_expectiles = {tau: [] for tau in tau_levels}

        for (real_returns,) in dataloader:
            real_returns = real_returns.to(device)
            current_batch_size = real_returns.size(0)

            # ----- Discriminator step -----
            optimizer_D.zero_grad()

            z = torch.randn(current_batch_size, latent_dim, device=device)
            fake_returns = generator(z)

            fake_estimates = discriminator(fake_returns.detach())
            real_estimates = discriminator(real_returns)

            score_fake = scorer(fake_estimates, real_returns)
            score_real = scorer(real_estimates, real_returns)

            # Discriminator maximizes score_fake - lambda * score_real, which (equivalent to minimizing the negative)
            d_loss = -(score_fake - lambda_dual * score_real)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_D.step()

            # ----- Generator step -----
            optimizer_G.zero_grad()

            z_new = torch.randn(current_batch_size, latent_dim, device=device)
            fake_returns_new = generator(z_new)
            fake_estimates_new = discriminator(fake_returns_new)

            g_loss = scorer(fake_estimates_new, real_returns)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer_G.step()

            # Track metrics
            epoch_d_loss.append(d_loss.item())
            epoch_g_loss.append(g_loss.item())

            # Denormalize expectile estimates for tracking
            with torch.no_grad():
                for i, tau in enumerate(tau_levels):
                    est_norm = fake_estimates[:, i].mean().item()
                    est_denorm = est_norm * data_std + data_mean
                    epoch_expectiles[tau].append(est_denorm)

        # Epoch averages
        history['d_loss'].append(np.mean(epoch_d_loss))
        history['g_loss'].append(np.mean(epoch_g_loss))
        for tau in tau_levels:
            history['expectile_estimates'][tau].append(np.mean(epoch_expectiles[tau]))

        if verbose and (epoch + 1) % 50 == 0:
            msg = f"Epoch [{epoch+1}/{n_epochs}] D_loss: {history['d_loss'][-1]:.6f} G_loss: {history['g_loss'][-1]:.6f}"
            for tau in tau_levels:
                msg += f" | e_{tau}: {history['expectile_estimates'][tau][-1]:.6f}"
            print(msg)

    # Wrap generator with denormalization
    class DenormalizedGenerator:
        def __init__(self, gen, mean, std):
            self.gen = gen
            self.mean = mean
            self.std = std

        def generate(self, n_samples, device=None):
            normalized = self.gen.generate(n_samples, device)
            return normalized * self.std + self.mean

        def eval(self):
            self.gen.eval()

    denorm_gen = DenormalizedGenerator(generator, data_mean, data_std)
    return denorm_gen, discriminator, history


def generate_gaussian_data(n_samples=1000, n_timesteps=50, mean=0.0, std=0.02):
    return np.random.normal(mean, std, (n_samples, n_timesteps)).astype(np.float32)

def calculate_var(returns, alpha=0.05):
    return np.percentile(returns.flatten(), alpha * 100)

def calculate_es(returns, alpha=0.05):
    var = calculate_var(returns, alpha)
    tail = returns.flatten()[returns.flatten() <= var]
    return tail.mean() if len(tail) > 0 else var


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Expectile-GAN: Standalone Test on Synthetic Gaussian Data")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 1000
    n_timesteps = 50
    std = 0.02

    data = generate_gaussian_data(n_samples, n_timesteps, std=std)
    print(f"\nData shape: {data.shape}")
    print(f"Data mean:  {data.mean():.6f}")
    print(f"Data std:   {data.std():.6f}")

    tau_levels = [0.01, 0.05, 0.1]

    # Compute ground truth expectiles
    print("\nGround truth expectiles (from data):")
    for tau in tau_levels:
        e = compute_expectile(data, tau)
        print(f"  e_{tau}: {e:.6f}")

    print(f"\nGround truth VaR(5%): {calculate_var(data, 0.05):.6f}")
    print(f"Ground truth ES(5%):  {calculate_es(data, 0.05):.6f}")

    # Train
    print("\n" + "-" * 70)
    print("Training Expectile-GAN...")
    print("-" * 70)

    gen, disc, history = train_expectile_gan(
        data,
        n_epochs=500,
        batch_size=64,
        latent_dim=100,
        lr_d=1e-5,
        lr_g=1e-5,
        lambda_dual=1.0,
        tau_levels=tau_levels,
        verbose=True,
    )

    # Generate and compare
    samples = gen.generate(n_samples)
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Real':>12} {'Expectile-GAN':>15}")
    print("-" * 52)
    print(f"{'Mean':<25} {data.mean():>12.6f} {samples.mean():>15.6f}")
    print(f"{'Std':<25} {data.std():>12.6f} {samples.std():>15.6f}")

    for tau in tau_levels:
        real_e = compute_expectile(data, tau)
        gen_e = compute_expectile(samples, tau)
        err = abs(gen_e - real_e) / abs(real_e) * 100
        print(f"{'Expectile ('+str(int(tau*100))+'%)':<25} {real_e:>12.6f} {gen_e:>15.6f}  ({err:.1f}%)")

    real_var = calculate_var(data, 0.05)
    gen_var = calculate_var(samples, 0.05)
    var_err = abs(gen_var - real_var) / abs(real_var) * 100
    print(f"{'VaR (5%)':<25} {real_var:>12.6f} {gen_var:>15.6f}  ({var_err:.1f}%)")

    real_es = calculate_es(data, 0.05)
    gen_es = calculate_es(samples, 0.05)
    es_err = abs(gen_es - real_es) / abs(real_es) * 100
    print(f"{'ES (5%)':<25} {real_es:>12.6f} {gen_es:>15.6f}  ({es_err:.1f}%)")
