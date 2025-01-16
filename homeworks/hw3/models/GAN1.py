import torch
from torch import nn
import torch.nn.functional as F


class GAN1Generator(nn.Module):
    def __init__(self, latent_dim=1, output_dim=1, hidden_dim=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.model.forward(z)

    def generate(self, batch_size, device):
        with torch.no_grad():
            z = torch.randn((batch_size, self.latent_dim)).to(device)
            x_fake = self.forward(z)
        return x_fake


class GAN1Discriminator(nn.Module):
    def __init__(self, latent_dim=1, output_dim=1, hidden_dim=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model.forward(x)
