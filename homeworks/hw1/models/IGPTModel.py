import torch
from torch import nn
from torch.nn import functional as F

from layer.MaskedAttentionLayer import MaskedAttentionLayer


class IGPTModel(nn.Module):
    def __init__(self, vocab_size, max_sequence_length, n_heads=4, embed_dim=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sequence_length = max_sequence_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, embed_dim)
        self.module = nn.Sequential(
            MaskedAttentionLayer(embed_dim=embed_dim, n_heads=n_heads),
        )
        exp = 2
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * exp),
            nn.GELU(),
            nn.Linear(embed_dim * exp, embed_dim * exp),
            nn.GELU(),
            nn.Linear(embed_dim * exp, vocab_size),
        )

    def loss_function(self, logits, target):
        logits = logits.permute(0, 2, 1)
        target = target[:, 1:]
        logits = logits[:, :, :-1]
        return F.cross_entropy(logits, target)

    def forward(self, x):
        f1 = self.token_embedding.forward(x) + self.position_embedding(torch.arange(0, x.shape[1], device=x.device))
        f2 = self.module(f1)
        f3 = self.linear.forward(f1 + f2)

        return f3

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

        return images[:, 1:]
