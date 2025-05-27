
import torch
import torch.nn as nn

class DiffDMC(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        return self.linear(x)
