import math

import torch
from torch import nn
from torch.nn import functional as F

from layer.MaskedAttentionLayer import MaskedAttentionLayer


class ColorIGPTModel(nn.Module):
    def __init__(self, vocab_size, max_sequence_length, n_heads=4, embed_dim=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, embed_dim)
        self.module = nn.Sequential(
            MaskedAttentionLayer(embed_dim=embed_dim, n_heads=n_heads),
        )
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_size),
        )

    def loss_function(self, logits, target):
        logits = logits.permute(0, 3, 1, 2)
        target = target[:, 1:]
        logits = logits[:, :, :-1]
        return F.cross_entropy(logits, target)

    def forward(self, x):
        xr = x[:, 0]
        xg = x[:, 1]
        xb = x[:, 2]
        embedding_r = self.token_embedding.forward(xr) + self.position_embedding(torch.arange(0, xr.shape[-1], device=x.device))
        embedding_g = self.token_embedding.forward(xg) + self.position_embedding(torch.arange(0, xg.shape[-1], device=x.device))
        embedding_b = self.token_embedding.forward(xb) + self.position_embedding(torch.arange(0, xb.shape[-1], device=x.device))
        f1_r = self.module(embedding_r)
        f1_g = self.module(embedding_g)
        f1_b = self.module(embedding_b)
        f3_r = self.linear.forward(f1_r)
        f3_g = self.linear.forward(f1_g)
        f3_b = self.linear.forward(f1_b)
        return torch.stack((f3_r, f3_g, f3_b), dim=1)

    def generate(self, batch_size=1, device="cpu"):
        images = torch.zeros(size=(batch_size, 1)).to(device).long()
        images[:, 0] = 2
        with torch.no_grad():
            for i in range(self.max_sequence_length):
                if i == 0:
                    continue
                next_logits = self.forward(images)

                next_probs = torch.softmax(next_logits[:, -1, :-1], dim=1)
                next_pixels = torch.multinomial(next_probs, num_samples=1)
                images = torch.cat((images, next_pixels), dim=1)

        return images[:, 1:].reshape(batch_size, 1, 20, 20)
