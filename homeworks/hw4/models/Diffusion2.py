import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.l_h1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
        )
        self.l_t1 = nn.Linear(temb_channels, out_channels)
        self.l_h2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
        )
        self.l_x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, temb):
        h = self.l_h1(x)
        temb = self.l_t1(temb)
        h += temb.unsqueeze(-1).unsqueeze(-1)  # h is BxDxHxW, temb is BxDx1x1
        h = self.l_h2(h)
        if self.in_channels != self.out_channels:
            x = self.l_x1(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.module(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        return self.module(x)


class Diffusion2(nn.Module):
    def __init__(self, in_channels, hidden_dims, blocks_per_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.blocks_per_dim = blocks_per_dim
        temb_channels = self.hidden_dims[0] * 4
        self.l_emb1 = nn.Sequential(
            nn.Linear(self.hidden_dims[0], temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels)
        )

        self.l_h1 = nn.Conv2d(self.in_channels, self.hidden_dims[0], 3, padding=1)
        self.res_blocks_down = nn.ModuleList([])
        self.downsample_blocks = nn.ModuleList([])
        prev_ch = self.hidden_dims[0]
        down_block_chans = [prev_ch]

        for i, hidden_dim in enumerate(self.hidden_dims):
            for _ in range(self.blocks_per_dim):
                self.res_blocks_down.append(ResidualBlock(prev_ch, hidden_dim, temb_channels))
                prev_ch = hidden_dim
                down_block_chans.append(prev_ch)
            if i != len(self.hidden_dims) - 1:
                self.downsample_blocks.append(Downsample(prev_ch))
                down_block_chans.append(prev_ch)
        self.res_blocks_mid = nn.ModuleList([
            ResidualBlock(prev_ch, prev_ch, temb_channels),
            ResidualBlock(prev_ch, prev_ch, temb_channels),
        ])

        self.res_blocks_up = nn.ModuleList([])
        self.upsample_blocks = nn.ModuleList([])
        for i, hidden_dim in list(enumerate(self.hidden_dims))[::-1]:
            for j in range(self.blocks_per_dim + 1):
                dch = down_block_chans.pop()
                self.res_blocks_up.append(ResidualBlock(prev_ch + dch, hidden_dim, temb_channels))
                prev_ch = hidden_dim
                if i and j == self.blocks_per_dim:
                    self.upsample_blocks.append(Upsample(prev_ch))
        self.l_out = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, self.in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        emb = self.timestep_embedding(t, self.hidden_dims[0])
        emb = self.l_emb1(emb)

        h = self.l_h1(x)
        hs = [h]
        r = 0
        d = 0
        for i, hidden_dim in enumerate(self.hidden_dims):
            for _ in range(self.blocks_per_dim):
                h = self.res_blocks_down[r](h, emb)
                r += 1
                hs.append(h)
            if i != len(self.hidden_dims) - 1:
                h = self.downsample_blocks[d](h)
                d += 1
                hs.append(h)

        for block in self.res_blocks_mid:
            h = block(h, emb)

        r = 0
        u = 0
        for i, hidden_dim in list(enumerate(self.hidden_dims))[::-1]:
            for j in range(self.blocks_per_dim + 1):
                h = self.res_blocks_up[r](torch.cat((h, hs.pop()), dim=1), emb)
                r += 1
                if i and j == self.blocks_per_dim:
                    h = self.upsample_blocks[u](h)
                    u += 1

        out = self.l_out(h)
        return out

    def loss_function(self, input, target):
        return F.mse_loss(input, target)

    @staticmethod
    def timestep_embedding(timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.tensor(np.exp(-np.log(max_period) * np.arange(0, half) / half), dtype=torch.float32).to(timesteps.device)
        args = timesteps[:, None] * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), axis=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1], dtype=torch.float32)), axis=-1)
        return embedding

    @staticmethod
    def generate_noise_steps(x, device):
        t = torch.rand((x.shape[0], 1, 1, 1)).to(device)
        alpha_t = torch.cos(torch.pi / 2 * t)
        sigma_t = torch.sin(torch.pi / 2 * t)
        epsilon = torch.randn_like(x).to(device)
        x_t = alpha_t * x + sigma_t * epsilon
        t = t.squeeze()
        return t, x_t, epsilon

    def generate(self, batch_size, num_steps, device):
        self.eval()
        with torch.no_grad():
            ts = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1).unsqueeze(-1)
            x_t = torch.randn(batch_size, self.in_channels, 32, 32).to(device)
            for i in range(num_steps):
                t = ts[i].tile(batch_size).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
                t_m1 = ts[i + 1].tile(batch_size).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
                alpha_t_m1 = torch.cos(torch.pi / 2 * t_m1)
                sigma_t_m1 = torch.sin(torch.pi / 2 * t_m1)
                alpha_t = torch.cos(torch.pi / 2 * t)
                sigma_t = torch.sin(torch.pi / 2 * t)
                eta_t = (sigma_t_m1 / sigma_t) * torch.sqrt((1 - (torch.pow(alpha_t, 2)) / torch.pow(alpha_t_m1, 2)))
                epsilon_hat = self.forward(x_t, t.squeeze())
                epsilon_t = torch.randn_like(x_t).to(device)
                x_t = (alpha_t_m1 * torch.clip((x_t - sigma_t * epsilon_hat) / alpha_t, -1, 1) +
                       torch.sqrt(torch.clip(torch.pow(sigma_t_m1, 2) - torch.pow(eta_t, 2), min=0)) * epsilon_hat +
                       eta_t * epsilon_t)

        return x_t
