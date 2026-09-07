import torch
import torch.nn as nn


class PINN(nn.Module):
    """MLP mapping (x, t) -> u(x, t)."""

    def __init__(self, hidden_dim: int = 32, hidden_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))
