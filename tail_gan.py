"""
Tail-GAN Implementation following Algorithm 4 from the thesis..

With the Fissler-Ziegel scoring function:

S_α(v, e; y) = (1{y ≤ v} - α)H_1(v) + (1/α)H_2(e)((v-y)1{y ≤ v} - α(v-e))

With the specific choices from the thesis:
    H_1(v) = -5v²
    H_2(e) = (α/2)e²

For α < 0.5 (left tail), this scoring function is strictly consistent
for the pair (VaR_α, ES_α).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Generator(nn.Module):
    """Generator network for Tail-GAN."""

    def __init__(self, latent_dim=100, output_dim=50):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

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
        """Generate n_samples of synthetic returns."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            samples = self.forward(z)
        return samples.cpu().numpy()


class TailDiscriminator(nn.Module):
    """
    Tail-GAN Discriminator that outputs (VaR, ES) estimates.

    The discriminator takes a batch of returns and outputs (v, e) pairs
    where v estimates VaR_α and e estimates ES_α.
    """

    def __init__(self, input_dim=50, alphas=[0.01 ,0.05]):
        super(TailDiscriminator, self).__init__()

        self.alphas = alphas
        self.n_outputs = 2 * len(alphas)  # (VaR, ES) for each alpha

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
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, returns):
        """
        Args:
            returns: Tensor of shape (batch_size, n_timesteps)

        Returns:
            Tensor of shape (batch_size, 2*len(alphas)) containing (VaR, ES) pairs
        """
        return self.model(returns)


def G1(v):
    """
    G_1(v) = H_1(v) = -5v².

    This corresponds to the quadratic choice for strictly consistent scoring.
    """
    return -5.0 * v ** 2


def G2(e, alpha):
    """
    G_2(e) = H_2(e) = (α/2)e².
    """
    return (alpha / 2.0) * e ** 2


def fissler_ziegel_score(v, e, y, alpha):
    """
    Strictly consistent scoring function for (VaR_α, ES_α) from Fissler-Ziegel.

    S_α(v, e; y) = (1{y ≤ v} - α)H_1(v) + (1/α)H_2(e)((v-y)1{y ≤ v} - α(v-e))

    With:
        H_1(v) = G_1(v) = -5v²
        H_2(e) = G_2(e) = (α/2)e²

    Args:
        v: VaR estimate (scalar or tensor)
        e: ES estimate (scalar or tensor)
        y: Actual returns/observations (tensor)
        alpha: Quantile level (e.g., 0.05 for 5% VaR)

    Returns:
        Scalar mean score value
    """
    # Indicator: 1{y ≤ v}
    indicator = (y <= v).float()

    # Term 1: (1{y ≤ v} - α) * G_1(v)
    # G_1(v) = -5v²
    term1 = (indicator - alpha) * G1(v)

    # Term 2: (1/α) * G_2(e) * ((v - y) * 1{y ≤ v} - α(v - e))
    # G_2(e) = (α/2)e²
    g2_e = G2(e, alpha)
    inner = (v - y) * indicator - alpha * (v - e)
    term2 = (1.0 / alpha) * g2_e * inner

    # Total score
    score = term1 + term2

    return score.mean()


class FisslerZiegelScorer(nn.Module):
    """
    Scorer module implementing the exact Fissler-Ziegel scoring function.

    S_α(v, e; y) = (1{y ≤ v} - α)H_1(v) + (1/α)H_2(e)((v-y)1{y ≤ v} - α(v-e))

    With H_1(v) = -5v², H_2(e) = (α/2)e²
    """

    def __init__(self, alphas=[0.01, 0.05]):
        super(FisslerZiegelScorer, self).__init__()
        self.alphas = alphas

    def forward(self, var_es_estimates, returns):
        """
        Compute the Fissler-Ziegel scoring function.

        Args:
            var_es_estimates: Tensor of shape (batch_size, 2*len(alphas))
                              containing (VaR, ES) pairs
            returns: Tensor of shape (batch_size, n_timesteps)

        Returns:
            Scalar score value
        """
        total_score = 0.0
        y = returns.flatten()  # All returns as observations

        for i, alpha in enumerate(self.alphas):
            # Extract VaR and ES estimates (take mean across batch)
            v = var_es_estimates[:, 2 * i].mean()       # VaR estimate
            e = var_es_estimates[:, 2 * i + 1].mean()   # ES estimate

            # Compute Fissler-Ziegel score
            score = fissler_ziegel_score(v, e, y, alpha)
            total_score = total_score + score

        return total_score


def train_tailgan(data, n_epochs=500, batch_size=64, latent_dim=100,
                  lr_d=1e-5, lr_g=1e-5, lambda_dual=1.0,
                  alphas=[0.01, 0.05], verbose=True):
    """
    Train Tail-GAN following Algorithm 4 from the thesis.

    Args:
        data: np.ndarray of shape (n_samples, n_timesteps) - financial returns
        n_epochs: Number of training epochs
        batch_size: Batch size (N_B)
        latent_dim: Dimension of latent space
        lr_d: Learning rate for discriminator
        lr_g: Learning rate for generator
        lambda_dual: Dual parameter
        alphas: List of quantile levels
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_generator, trained_discriminator, loss_history)
    """
    device = get_device()
    if verbose:
        print(f"Training Tail-GAN on {device}")
        print(f"Parameters: lr_d={lr_d}, lr_g={lr_g}, lambda={lambda_dual}, alphas={alphas}")

    output_dim = data.shape[1]

    data_mean = data.mean()
    data_std = data.std()
    data_normalized = (data - data_mean) / data_std

    # Initialize models
    generator = Generator(latent_dim=latent_dim, output_dim=output_dim).to(device)
    discriminator = TailDiscriminator(input_dim=output_dim, alphas=alphas).to(device)
    scorer = FisslerZiegelScorer(alphas=alphas)

    # Optimizers with lower learning rates for stability
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # Data loader
    tensor_data = torch.FloatTensor(data_normalized)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Loss history
    history = {'d_loss': [], 'g_loss': [], 'var_estimate': [], 'es_estimate': []}

    for epoch in range(n_epochs):
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_var = []
        epoch_es = []

        for real_returns, in dataloader:
            real_returns = real_returns.to(device)
            current_batch_size = real_returns.size(0)

            # =====================================================
            # Step 3-4: Generate noise and sample batch
            # =====================================================
            z = torch.randn(current_batch_size, latent_dim, device=device)
            fake_returns = generator(z)

            # =====================================================
            # Train Discriminator
            # L_D = S(D(fake), real) - λ * S(D(real), real)
            # Discriminator maximizes this (gradient ascent)
            # =====================================================
            optimizer_D.zero_grad()

            fake_estimates = discriminator(fake_returns.detach())
            real_estimates = discriminator(real_returns)

            # Score on fake returns (compared to real distribution)
            score_fake = scorer(fake_estimates, real_returns)
            # Score on real returns
            score_real = scorer(real_estimates, real_returns)

            # Discriminator loss: maximize score_fake - λ*score_real
            # We minimize the negative
            d_loss = -(score_fake - lambda_dual * score_real)

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_D.step()

            epoch_d_loss.append(d_loss.item())

            # =====================================================
            # Train Generator
            # L_G = S(D(fake), real)
            # Generator minimizes this
            # =====================================================
            optimizer_G.zero_grad()

            z_new = torch.randn(current_batch_size, latent_dim, device=device)
            fake_returns_new = generator(z_new)
            fake_estimates_new = discriminator(fake_returns_new)

            # Generator loss: minimize score
            g_loss = scorer(fake_estimates_new, real_returns)

            g_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer_G.step()

            epoch_g_loss.append(g_loss.item())

            # Track VaR/ES estimates (denormalized)
            with torch.no_grad():
                var_norm = fake_estimates[:, 0].mean().item()
                es_norm = fake_estimates[:, 1].mean().item()
                # Denormalize
                epoch_var.append(var_norm * data_std + data_mean)
                epoch_es.append(es_norm * data_std + data_mean)

        # Record epoch statistics
        history['d_loss'].append(np.mean(epoch_d_loss))
        history['g_loss'].append(np.mean(epoch_g_loss))
        history['var_estimate'].append(np.mean(epoch_var))
        history['es_estimate'].append(np.mean(epoch_es))

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}] "
                  f"D_loss: {history['d_loss'][-1]:.4f} "
                  f"G_loss: {history['g_loss'][-1]:.4f} "
                  f"VaR_est: {history['var_estimate'][-1]:.6f} "
                  f"ES_est: {history['es_estimate'][-1]:.6f}")

    # Create a wrapper generator that denormalizes output
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

    denorm_generator = DenormalizedGenerator(generator, data_mean, data_std)

    return denorm_generator, discriminator, history


# =============================================================================
#  Other Tail-GAN training
# =============================================================================
# Was created as a test to look at other learning rates and to use the n_critic:
# the number of discriminator updates per generator update.
# =============================================================================

def train_tailgan_improved(
    data,
    n_epochs=500,
    batch_size=64,
    latent_dim=100,
    lr_d=1e-5,
    lr_g=1e-5,
    lambda_dual=1.0,
    alphas=(0.01, 0.05),
    n_critic=5,
    verbose=True,
):
    """
    Improved Tail-GAN training.

      * `n_critic`  - the discriminator is updated this many times per
                      generator update (default 5, like WGAN-GP).

    Returns
    -------
    denorm_generator : object with .generate(n_samples)
    discriminator    : trained TailDiscriminator
    history          : dict with keys 'd_loss', 'g_loss',
                       'var_estimate' (dict alpha -> list),
                       'es_estimate'  (dict alpha -> list)
    """
    device = get_device()
    alphas = list(alphas)
    output_dim = data.shape[1]

    if verbose:
        print(f"  [train_tailgan_improved]  device={device}")
        print(f"  lr_d={lr_d}, lr_g={lr_g}  "
              f"(TTUR ratio {lr_d/lr_g:.2f}x),  n_critic={n_critic}")
        print(f"  lambda_dual={lambda_dual}, alphas={alphas}")

    data_mean = float(data.mean())
    data_std  = float(data.std())
    data_norm = (data - data_mean) / data_std

    generator     = Generator(latent_dim=latent_dim,
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
            bs = real.size(0)

            # n_critic discriminator updates per generator update
            for _ in range(n_critic):
                opt_D.zero_grad()
                with torch.no_grad():
                    z = torch.randn(bs, latent_dim, device=device)
                    fake = generator(z)
                fake_est = discriminator(fake)
                real_est = discriminator(real)
                s_fake = scorer(fake_est, real)
                s_real = scorer(real_est, real)
                d_loss = -(s_fake - lambda_dual * s_real)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(), 1.0
                )
                opt_D.step()
            d_buf.append(d_loss.item())

            # single generator update with the Fissler-Ziegel score only once
            opt_G.zero_grad()
            z2 = torch.randn(bs, latent_dim, device=device)
            fake2 = generator(z2)
            g_loss = scorer(discriminator(fake2), real)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_G.step()
            g_buf.append(g_loss.item())

            with torch.no_grad():
                est = discriminator(fake2.detach())
                for i, a in enumerate(alphas):
                    v = est[:, 2 * i    ].mean().item() * data_std + data_mean
                    e = est[:, 2 * i + 1].mean().item() * data_std + data_mean
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
            print(f"  [{epoch+1}/{n_epochs}]  "
                  f"D:{history['d_loss'][-1]:.4f}  "
                  f"G:{history['g_loss'][-1]:.4f}  {parts}")

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

    return (_DenormGen(generator, data_mean, data_std),
            discriminator,
            history)


def generate_gaussian_data(n_samples=1000, n_timesteps=50, mean=0.0, std=0.02):
    """Generate synthetic Gaussian returns data."""
    return np.random.normal(mean, std, (n_samples, n_timesteps)).astype(np.float32)


def calculate_var(returns, alpha=0.05):
    """Calculate empirical VaR at alpha percentile."""
    return np.percentile(returns.flatten(), alpha * 100)


def calculate_es(returns, alpha=0.05):
    """Calculate empirical ES (Expected Shortfall)."""
    var = calculate_var(returns, alpha)
    tail = returns.flatten()[returns.flatten() <= var]
    return tail.mean() if len(tail) > 0 else var


if __name__ == "__main__":
    print("=" * 60)
    print("Tail-GAN Training")
    print("Scoring function: Fissler-Ziegel")
    print("=" * 60)

    # Generate synthetic data
    print("\nGenerating synthetic Gaussian data...")
    data = generate_gaussian_data(n_samples=1000, n_timesteps=50, std=0.02)
    print(f"Data shape: {data.shape}")
    print(f"Data stats - Mean: {data.mean():.6f}, Std: {data.std():.6f}")

    # Calculate true VaR and ES
    alpha = 0.02
    true_var = calculate_var(data, alpha)
    true_es = calculate_es(data, alpha)
    print(f"\nTrue VaR ({alpha*100:.0f}%): {true_var:.6f}")
    print(f"True ES ({alpha*100:.0f}%):  {true_es:.6f}")

    # Train Tail-GAN
    print("\nTraining Tail-GAN...")
    generator, discriminator, history = train_tailgan(
        data,
        n_epochs=500,
        batch_size=64,
        latent_dim=100,
        lr_d=1e-5,
        lr_g=1e-5,
        lambda_dual=1.0,
        alphas=[0.01, 0.05],
        verbose=True
    )

    # Generate synthetic samples
    print("\nGenerating synthetic samples...")
    fake_data = generator.generate(1000)
    print(f"Generated data shape: {fake_data.shape}")

    # Compare statistics
    print("\n" + "=" * 60)
    print("Statistics Comparison")
    print("=" * 60)
    print(f"Real data    - Mean: {data.mean():.6f}, Std: {data.std():.6f}")
    print(f"Generated    - Mean: {fake_data.mean():.6f}, Std: {fake_data.std():.6f}")

    # Tail risk comparison
    fake_var = calculate_var(fake_data, alpha)
    fake_es = calculate_es(fake_data, alpha)

    print(f"\nVaR ({alpha*100:.0f}%)     - Real: {true_var:.6f}, Generated: {fake_var:.6f}")
    print(f"ES ({alpha*100:.0f}%)      - Real: {true_es:.6f}, Generated: {fake_es:.6f}")

    print(f"\nVaR Error: {abs(true_var - fake_var):.6f} ({abs(true_var - fake_var)/abs(true_var)*100:.2f}%)")
    print(f"ES Error:  {abs(true_es - fake_es):.6f} ({abs(true_es - fake_es)/abs(true_es)*100:.2f}%)")
