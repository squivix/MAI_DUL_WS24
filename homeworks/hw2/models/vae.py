import torch
from torch import nn

from torch.nn import functional as F


class VAEModel(nn.Module):
    def __init__(self, input_size, latent_size, hidden_units, regularization_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.latent_size = latent_size
        self.beta = regularization_weight

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, latent_size * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, input_size * 2),
        )

    def forward(self, x):
        encoder_logits = self.encoder.forward(x)
        mu_z = encoder_logits[:, :self.latent_size]
        log_var_z = encoder_logits[:, self.latent_size:]
        sigma_z = torch.exp(0.5 * log_var_z)
        z = sigma_z * torch.randn_like(sigma_z) + mu_z

        decoder_logits = self.decoder.forward(z)
        mu_x_fake = decoder_logits[:, :self.input_size]
        log_var_x_fake = decoder_logits[:, self.input_size:]
        sigma_x_fake = torch.exp(0.5 * log_var_x_fake)
        x_fake = sigma_x_fake * torch.randn_like(sigma_x_fake) + mu_x_fake
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

    def generate(self, n_examples, with_noise=True):
        z = torch.randn(n_examples, self.latent_size).to(self.device)
        decoder_logits = self.decoder.forward(z)
        mu_x_fake = decoder_logits[:, :self.input_size]
        log_var_x_fake = decoder_logits[:, self.input_size:]
        sigma_x_fake = torch.exp(0.5 * log_var_x_fake)

        if not with_noise:
            return mu_x_fake
        x_fake = sigma_x_fake * torch.randn_like(sigma_x_fake) + mu_x_fake
        return x_fake
