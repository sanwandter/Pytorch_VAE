import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    """
    A small convolutional VAE for 1x28x28 images.
    Forward returns: recon (same shape as input), mu, log_var
    """
    def __init__(self, z_dim=128):
        super().__init__()

        # Encoder adapted for MNIST: 1 x 28 x 28 -> feature map 32 x 7 x 7
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # -> 32 x 14 x 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), # -> 64 x 7 x 7
            nn.ReLU(True),
            nn.Conv2d(64, 32, 1, 1, 0), # -> 32 x 7 x 7 (keep spatial)
            nn.ReLU(True),
            nn.Conv2d(32, 32, 1, 1, 0), # -> 32 x 7 x 7
            nn.ReLU(True),
        )

        self.feature_dim = 32 * 7 * 7
        self.fc_mu = nn.Linear(self.feature_dim, z_dim)
        self.fc_log_var = nn.Linear(self.feature_dim, z_dim)

        # Decoder: project back from feature map 32 x 7 x 7 -> 1 x 28 x 28
        self.fc_dec = nn.Linear(z_dim, self.feature_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, 1, 1),   # keep 7 x 7
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),   # -> 16 x 14 x 14
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),    # -> 8 x 28 x 28
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, 1, 1),     # -> 1 x 28 x 28
            nn.Sigmoid(),
        )

    def encode(self, x):
        # Encode input to latent space q_phi(z|x)
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def decode(self, z):
        # Decode latent variable to reconstruct input p_theta(x|z)
        h = self.fc_dec(z)
        h = h.view(h.size(0), 32, 7, 7)
        x_recon = self.dec(h)
        return x_recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        sigma = torch.sqrt(torch.exp(log_var))  # Get standard deviation from log variance
        eps = torch.randn_like(sigma) # Epsilon from N(0,1)
        z = mu + eps * sigma
        x_recon = self.decode(z)
        return x_recon, mu, log_var


if __name__ == '__main__':
    # quick smoke test
    model = VariationalAutoEncoder(z_dim=128)
    x = torch.randn(2, 1, 28, 28)
    x_recon, mu, log_var = model(x)
    print('recon', x_recon.shape, 'mu', mu.shape, 'log_var', log_var.shape)

