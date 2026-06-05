"""
WGAN-GP Implementation following Algorithm 3 from the thesis.

Key parameters from the paper:
- (lambda_gp) = 10: Gradient penalty coefficient
- n_critic = 5: Number of critic iterations per generator iteration
- (lr) = 0.0001: Learning rate
- beta_1 = 0, beta_2 = 0.9: Adam hyperparameters

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
    """Generator network for WGAN-GP."""

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
        """Generate n_samples of synthetic data."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            samples = self.forward(z)
        return samples.cpu().numpy()


class Critic(nn.Module):
    """
    Critic (Discriminator) network for WGAN-GP.

    Note: In WGAN-GP, we do NOT use batch normalization in the critic
    because it can interfere with the gradient penalty calculation.
    """

    def __init__(self, input_dim=50):
        super(Critic, self).__init__()

        # No batch normalization in critic for WGAN-GP!
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """
    Compute gradient penalty for WGAN-GP.
    """
    batch_size = real_samples.size(0)

    # Sample epsilon ~ Uniform[0, 1]
    epsilon = torch.rand(batch_size, 1, device=device)

    interpolated = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)

    # D(x̂)
    d_interpolated = critic(interpolated)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


def train_wgan_gp(data, n_epochs=300, batch_size=64, latent_dim=100,
                  lr=1e-4, n_critic=5, lambda_gp=10,
                  beta1=0.0, beta2=0.9, verbose=True):
    """
    Train WGAN-GP following Algorithm 3 from the thesis.

    Args:
        data: np.ndarray of shape (n_samples, n_timesteps) - financial returns
        n_epochs: Number of training epochs
        batch_size: Batch size (m)
        latent_dim: Dimension of latent space
        lr: Learning rate (α = 0.0001)
        n_critic: Number of critic updates per generator update (5)
        lambda_gp: Gradient penalty coefficient (λ = 10)
        beta1: Adam β1 (0)
        beta2: Adam β2 (0.9)
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_generator, loss_history)
    """
    device = get_device()
    if verbose:
        print(f"Training WGAN-GP on {device}")
        print(f"Parameters: lr={lr}, n_critic={n_critic}, lambda_gp={lambda_gp}, "
              f"beta1={beta1}, beta2={beta2}")

    output_dim = data.shape[1]
    n_samples = data.shape[0]

    # Initialize models
    generator = Generator(latent_dim=latent_dim, output_dim=output_dim).to(device)
    critic = Critic(input_dim=output_dim).to(device)

    # Optimizers with Adam parameters:
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))

    # Data loader
    tensor_data = torch.FloatTensor(data)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Loss history
    history = {'critic_loss': [], 'generator_loss': [], 'wasserstein_dist': [], 'gradient_penalty': []}

    for epoch in range(n_epochs):
        epoch_critic_loss = []
        epoch_gen_loss = []
        epoch_wasserstein = []
        epoch_gp = []

        for real_batch, in dataloader:
            real_batch = real_batch.to(device)
            current_batch_size = real_batch.size(0)

            # ==================================
            # Train Critic n_critic times
            # ==================================
            for _ in range(n_critic):
                optimizer_C.zero_grad()

                # Sample latent
                z = torch.randn(current_batch_size, latent_dim, device=device)

                # Generator on it
                fake_batch = generator(z).detach()

                # Critics on data
                critic_real = critic(real_batch)
                critic_fake = critic(fake_batch)

                # Gradient penalty
                gp = compute_gradient_penalty(critic, real_batch, fake_batch, device)

                critic_loss = critic_fake.mean() - critic_real.mean() + lambda_gp * gp

                # Update critic
                critic_loss.backward()
                optimizer_C.step()

            # Record metrics
            epoch_critic_loss.append(critic_loss.item())
            epoch_wasserstein.append((critic_real.mean() - critic_fake.mean()).item())
            epoch_gp.append(gp.item())

            # ==============================
            # Train Generator once
            # ==============================
            optimizer_G.zero_grad()

            # Sample new latent variables
            z = torch.randn(current_batch_size, latent_dim, device=device)

            # Generate fake samples
            fake_batch = generator(z)

            # Generator loss
            # Generator wants to maximize D(fake), so minimize -D(fake)
            gen_loss = -critic(fake_batch).mean()

            gen_loss.backward()
            optimizer_G.step()

            epoch_gen_loss.append(gen_loss.item())

        # Record epoch statistics
        history['critic_loss'].append(np.mean(epoch_critic_loss))
        history['generator_loss'].append(np.mean(epoch_gen_loss))
        history['wasserstein_dist'].append(np.mean(epoch_wasserstein))
        history['gradient_penalty'].append(np.mean(epoch_gp))

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}] "
                  f"C_loss: {history['critic_loss'][-1]:.4f} "
                  f"G_loss: {history['generator_loss'][-1]:.4f} "
                  f"W_dist: {history['wasserstein_dist'][-1]:.4f} "
                  f"GP: {history['gradient_penalty'][-1]:.4f}")

    return generator, history


def generate_gaussian_data(n_samples=1000, n_timesteps=50, mean=0.0, std=0.02):
    """Generate synthetic Gaussian returns data."""
    return np.random.normal(mean, std, (n_samples, n_timesteps)).astype(np.float32)


if __name__ == "__main__":
    print("=" * 60)
    print("WGAN-GP Training (Algorithm 2 from thesis)")
    print("=" * 60)

    # Generate synthetic data
    print("\nGenerating synthetic Gaussian data...")
    data = generate_gaussian_data(n_samples=1000, n_timesteps=50, std=0.02)
    print(f"Data shape: {data.shape}")
    print(f"Data stats - Mean: {data.mean():.6f}, Std: {data.std():.6f}")


    print("\nTraining WGAN-GP with paper parameters...")
    print("  lr=0.0001, n_critic=5, lambda_gp=10, beta1=0, beta2=0.9")

    generator, history = train_wgan_gp(
        data,
        n_epochs=300,
        batch_size=64,
        latent_dim=100,
        lr=1e-4,
        n_critic=5,
        lambda_gp=10,
        beta1=0.0,
        beta2=0.9,
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

    # Simple tail comparison
    alpha = 0.05
    real_var = np.percentile(data.flatten(), alpha * 100)
    fake_var = np.percentile(fake_data.flatten(), alpha * 100)
    real_es = data.flatten()[data.flatten() <= real_var].mean()
    fake_es = fake_data.flatten()[fake_data.flatten() <= fake_var].mean()

    print(f"\nVaR (5%)     - Real: {real_var:.6f}, Generated: {fake_var:.6f}")
    print(f"ES (5%)      - Real: {real_es:.6f}, Generated: {fake_es:.6f}")
