import torch
from torch import nn

from torch.nn import functional as F


class VAEModel2(nn.Module):
    def __init__(self, latent_size=16, regularization_weight=0.0001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_size = latent_size
        self.beta = regularization_weight

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 2 * latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 4 * 4 * 128),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_logits = self.encoder.forward(x)
        mu_z = encoder_logits[:, :self.latent_size]
        log_var_z = encoder_logits[:, self.latent_size:]
        sigma_z = torch.exp(0.5 * log_var_z)
        z = sigma_z * torch.randn_like(sigma_z) + mu_z

        x_fake = self.decoder.forward(z)
        return x_fake, mu_z, log_var_z

    def loss_function(self, x_true, model_output):
        x_fake, mu, log_var = model_output
        reconstruction_loss = F.mse_loss(x_fake, x_true)
        regularization_term = torch.mean(0.5 * torch.sum(mu ** 2 + log_var.exp() - log_var - 1, dim=1), dim=0)
        loss = (1 - self.beta) * reconstruction_loss + self.beta * regularization_term
        return loss, reconstruction_loss, regularization_term

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, n_examples):
        z = torch.randn(n_examples, self.latent_size).to(self.device)
        x_fake = self.decoder.forward(z)
        return x_fake
