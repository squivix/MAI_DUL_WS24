import torch
from torch import nn
from torch.nn import functional as F

from layer.MaskedConv2d import MaskedConv2d


class ColoredPixelCNN(nn.Module):
    def __init__(self, filters=64):
        super().__init__()
        self.model = nn.Sequential(
            MaskedConv2d(in_channels=3, out_channels=filters, kernel_size=7, padding="same", mask_type="A"),
            nn.ReLU(),
            MaskedConv2d(in_channels=filters, out_channels=filters, kernel_size=7, padding="same", mask_type="B"),
            nn.ReLU(),
            MaskedConv2d(in_channels=filters, out_channels=filters, kernel_size=7, padding="same", mask_type="B"),
            nn.ReLU(),
            MaskedConv2d(in_channels=filters, out_channels=filters, kernel_size=7, padding="same", mask_type="B"),
            nn.ReLU(),
            MaskedConv2d(in_channels=filters, out_channels=filters, kernel_size=7, padding="same", mask_type="B"),
            nn.ReLU(),
            MaskedConv2d(in_channels=filters, out_channels=filters, kernel_size=7, padding="same", mask_type="B"),
            nn.ReLU(),

            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=filters, out_channels=2, kernel_size=1),
        )

    def forward(self, x):
        return self.model(x)

    def loss_function(self, x, logits):
        return F.nll_loss(torch.log_softmax(logits, dim=1), (x[:, 0] == -1).long())

    def generate(self, batch_size=1, device="cpu"):
        width = 20
        height = 20
        images = torch.zeros(size=(batch_size, 1, 20, 20)).to(device)
        images[:, :, 0, 0] = torch.randint(0, 2, (batch_size, 1))
        with torch.no_grad():
            for i in range(width):
                for j in range(height):
                    if i == 0 and j == 0:
                        continue
                    next_logits = self.forward(images * 2 - 1)

                    next_probs = (torch.softmax(next_logits, dim=1)[:, 0]).float()
                    next_pixels = (torch.rand((batch_size,)) < next_probs[:, i, j].cpu()).int()[:, None]
                    images[:, :, i, j] = next_pixels
        return images
