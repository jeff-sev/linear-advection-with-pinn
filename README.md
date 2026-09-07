# Linear Advection with a Physics-Informed Neural Network (PINN)

Solves the 1D linear advection equation

    u_t + c * u_x = 0,      x in [0, 1], t in [0, 1]
    u(x, 0) = sin(2*pi*x)   (initial condition)
    u(0, t) = u(1, t)       (periodic boundary condition)

with a PINN: a small MLP `u_theta(x, t)` trained so that its automatic
derivatives satisfy the PDE residual, the initial condition, and the
periodic boundary condition simultaneously.

The exact solution is the traveling wave `u(x, t) = sin(2*pi*(x - c*t))`,
which is used only to measure error, not during training.

## Project layout

```
pinn/
  model.py     MLP network definition
  problem.py   exact solution + PDE/IC/BC residual losses
  train.py     training loop + CLI entry point
outputs/       generated plots and metrics (created on run)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m pinn.train
```

Useful flags: `--epochs`, `--lr`, `--c` (advection speed), `--n-collocation`,
`--seed`, `--outdir`. Run `python -m pinn.train --help` for the full list.

Training prints the PDE/IC/BC/total loss periodically and, on completion,
writes `outputs/loss_history.png` (training curves) and
`outputs/solution_comparison.png` (predicted vs. exact solution snapshots
plus the spatiotemporal error field) along with a `outputs/metrics.json`
summary (final losses and relative L2 error against the exact solution).
