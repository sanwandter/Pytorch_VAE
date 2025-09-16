import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    """
    A small convolutional VAE for  images.
    Forward returns: recon (same shape as input), mu, log_var
    """
    def __init__(self, z_dim=128):
        super().__init__()

        # Encoder for CelebA: 3 x 128 x 128 -> feature map 256 x 8 x 8
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # -> 32 x 64 x 64
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> 64 x 32 x 32
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # -> 128 x 16 x 16
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),# -> 256 x 8 x 8
            nn.ReLU(True),
        )

        self.feature_dim = 256 * 8 * 8
        self.fc_mu = nn.Linear(self.feature_dim, z_dim)
        self.fc_log_var = nn.Linear(self.feature_dim, z_dim)

        # Decoder: project back from feature map 256 x 8 x 8 -> 3 x 128 x 128
        self.fc_dec = nn.Linear(z_dim, self.feature_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # -> 128 x 16 x 16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # -> 64 x 32 x 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),     # -> 32 x 64 x 64
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),     # -> 16 x 128 x 128
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, 1, 1),      # -> 3 x 128 x 128
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
        h = h.view(h.size(0), 256, 8, 8)
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
    x = torch.randn(2, 3, 128, 128)
    x_recon, mu, log_var = model(x)
    print('recon', x_recon.shape, 'mu', mu.shape, 'log_var', log_var.shape)

