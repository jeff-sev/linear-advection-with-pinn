import math

import torch

from .model import PINN


def exact_solution(x: torch.Tensor, t: torch.Tensor, c: float) -> torch.Tensor:
    """u(x, t) = sin(2*pi*(x - c*t)), the exact traveling-wave solution."""
    return torch.sin(2 * math.pi * (x - c * t))


def pde_residual_loss(model: PINN, x: torch.Tensor, t: torch.Tensor, c: float) -> torch.Tensor:
    """Mean-squared residual of u_t + c*u_x = 0 at collocation points (x, t)."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)

    grad_outputs = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, grad_outputs=grad_outputs, create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=grad_outputs, create_graph=True)[0]

    residual = u_t + c * u_x
    return torch.mean(residual ** 2)


def initial_condition_loss(model: PINN, x_ic: torch.Tensor) -> torch.Tensor:
    """Enforces u(x, 0) = sin(2*pi*x)."""
    t0 = torch.zeros_like(x_ic)
    u_pred = model(x_ic, t0)
    u_true = torch.sin(2 * math.pi * x_ic)
    return torch.mean((u_pred - u_true) ** 2)


def periodic_boundary_loss(model: PINN, t_bc: torch.Tensor) -> torch.Tensor:
    """Enforces u(0, t) = u(1, t) (periodic boundary condition)."""
    x_left = torch.zeros_like(t_bc)
    x_right = torch.ones_like(t_bc)
    u_left = model(x_left, t_bc)
    u_right = model(x_right, t_bc)
    return torch.mean((u_left - u_right) ** 2)
