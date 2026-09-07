import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import PINN
from .problem import (
    exact_solution,
    initial_condition_loss,
    pde_residual_loss,
    periodic_boundary_loss,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train a PINN for 1D linear advection.")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--c", type=float, default=1.0, help="advection speed")
    p.add_argument("--n-collocation", type=int, default=2000)
    p.add_argument("--n-ic", type=int, default=200)
    p.add_argument("--n-bc", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--hidden-layers", type=int, default=4)
    p.add_argument("--ic-weight", type=float, default=10.0)
    p.add_argument("--bc-weight", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"epoch": [], "pde": [], "ic": [], "bc": [], "total": []}

    for epoch in range(1, args.epochs + 1):
        x_f = torch.rand(args.n_collocation, 1, device=device)
        t_f = torch.rand(args.n_collocation, 1, device=device)
        x_ic = torch.rand(args.n_ic, 1, device=device)
        t_bc = torch.rand(args.n_bc, 1, device=device)

        loss_pde = pde_residual_loss(model, x_f, t_f, args.c)
        loss_ic = initial_condition_loss(model, x_ic)
        loss_bc = periodic_boundary_loss(model, t_bc)
        loss = loss_pde + args.ic_weight * loss_ic + args.bc_weight * loss_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.log_every == 0 or epoch == 1:
            history["epoch"].append(epoch)
            history["pde"].append(loss_pde.item())
            history["ic"].append(loss_ic.item())
            history["bc"].append(loss_bc.item())
            history["total"].append(loss.item())
            print(
                f"epoch {epoch:5d} | pde {loss_pde.item():.3e} | "
                f"ic {loss_ic.item():.3e} | bc {loss_bc.item():.3e} | "
                f"total {loss.item():.3e}"
            )

    metrics = evaluate_and_plot(model, args, device, history)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def evaluate_and_plot(model, args, device, history):
    model.eval()

    nx, nt = 200, 200
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    x_flat = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    t_flat = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(nx, nt)
    u_exact = exact_solution(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32),
        args.c,
    ).numpy()

    rel_l2_error = float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))

    # Loss history plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(history["epoch"], history["pde"], label="PDE residual")
    ax.semilogy(history["epoch"], history["ic"], label="Initial condition")
    ax.semilogy(history["epoch"], history["bc"], label="Periodic BC")
    ax.semilogy(history["epoch"], history["total"], label="Total", linewidth=2, color="black")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "loss_history.png"), dpi=150)
    plt.close(fig)

    # Solution comparison plot: snapshots + error field
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for t_snap in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = int(round(t_snap * (nt - 1)))
        axes[0, 0].plot(x, u_exact[idx, :], "--", label=f"exact t={t_snap:.2f}")
        axes[0, 0].plot(x, u_pred[idx, :], label=f"pred t={t_snap:.2f}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("u")
    axes[0, 0].set_title("Predicted vs exact snapshots")
    axes[0, 0].legend(fontsize=7, ncol=2)

    im0 = axes[0, 1].pcolormesh(X, T, u_pred, shading="auto", cmap="RdBu_r")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("t")
    axes[0, 1].set_title("PINN prediction u(x,t)")
    fig.colorbar(im0, ax=axes[0, 1])

    im1 = axes[1, 0].pcolormesh(X, T, u_exact, shading="auto", cmap="RdBu_r")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("t")
    axes[1, 0].set_title("Exact solution u(x,t)")
    fig.colorbar(im1, ax=axes[1, 0])

    err = np.abs(u_pred - u_exact)
    im2 = axes[1, 1].pcolormesh(X, T, err, shading="auto", cmap="viridis")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("t")
    axes[1, 1].set_title(f"Absolute error (rel. L2 = {rel_l2_error:.3e})")
    fig.colorbar(im2, ax=axes[1, 1])

    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "solution_comparison.png"), dpi=150)
    plt.close(fig)

    return {
        "final_pde_loss": history["pde"][-1],
        "final_ic_loss": history["ic"][-1],
        "final_bc_loss": history["bc"][-1],
        "final_total_loss": history["total"][-1],
        "relative_l2_error": rel_l2_error,
    }


if __name__ == "__main__":
    main()
