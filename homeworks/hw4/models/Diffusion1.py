import torch
from sympy.abc import epsilon
from torch import nn
from torch.nn import functional as F


class Diffusion1(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    @staticmethod
    def generate_noise_steps(x, device):
        t = torch.rand((x.shape[0], 1)).to(device)
        alpha_t = torch.cos(torch.pi / 2 * t)
        sigma_t = torch.sin(torch.pi / 2 * t)
        epsilon = torch.randn_like(x).to(device)
        x_t = alpha_t * x + sigma_t * epsilon
        return t, x_t, epsilon

    def forward(self, x_t, t):
        return self.model.forward(torch.hstack((x_t, t)))

    def loss_function(self, input, target):
        return F.mse_loss(input, target)

    def generate(self, batch_size, num_steps, device):
        ts = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1).unsqueeze(-1).tile(batch_size).to(device)
        x_t = torch.randn(batch_size, self.input_size).to(device)
        for i in range(num_steps):
            t = ts[i].unsqueeze(-1)
            t_m1 = ts[i + 1].unsqueeze(-1)
            alpha_t_m1 = torch.cos(torch.pi / 2 * t_m1)
            sigma_t_m1 = torch.sin(torch.pi / 2 * t_m1)
            alpha_t = torch.cos(torch.pi / 2 * t)
            sigma_t = torch.sin(torch.pi / 2 * t)
            eta_t = (sigma_t_m1 / sigma_t) * torch.sqrt((1 - (torch.pow(alpha_t, 2)) / torch.pow(alpha_t_m1, 2)))
            epsilon_hat = self.forward(x_t, t)
            epsilon_t = torch.randn(batch_size, self.input_size).to(device)
            x_t = (alpha_t_m1 * ((x_t - sigma_t * epsilon_hat) / alpha_t) +
                   torch.sqrt(torch.clip(torch.pow(sigma_t_m1, 2) - torch.pow(eta_t, 2), min=0)) * epsilon_hat +
                   eta_t * epsilon_t)
        return x_t
