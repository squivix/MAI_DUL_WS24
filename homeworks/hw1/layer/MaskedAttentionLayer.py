import math

import torch
from torch import nn
from torch.nn import functional as F


class MaskedAttentionLayer(nn.Module):
    def __init__(self, embed_dim, max_sequence_length):
        super().__init__()
        self.embed_size = embed_dim

        # Define the three weight matrices for Q, K, and V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Scaling factor
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_size = x.shape

        # Linear transformations to get Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Create a mask to prevent attending to future tokens
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(scores.device)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)

        # Apply mask and softmax to get attention weights
        masked_scores = scores + mask
        attention = F.softmax(masked_scores, dim=-1)

        # Get the weighted sum of values
        out = torch.matmul(attention, V)

        return out
    # def __init__(self, embed_dim, sequence_length, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.q_weights = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
    #     self.k_weights = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
    #     self.v_weights = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
    #     self.d_k = embed_dim
    #     nn.init.kaiming_uniform_(self.q_weights)
    #     nn.init.kaiming_uniform_(self.k_weights)
    #     nn.init.kaiming_uniform_(self.v_weights)
    #
    # def forward(self, x):
    #     q = x @ self.q_weights.T
    #     k = x @ self.k_weights.T
    #     v = x @ self.v_weights.T
    #     sequence_length = x.shape[1]
    #     mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(x.device)
    #
    #     s = q @ k.permute(0, 2, 1) / math.sqrt(self.d_k)
    #     # - (1 - mask) * 1e10
    #     s = torch.softmax(s, dim=1)
    #     logits = s @ v
    #     return logits
