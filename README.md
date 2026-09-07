# Linear Advection: Fortran90 vs. Python CFD, and a PINN

Solves the 1D linear advection equation

    u_t + c * u_x = 0,      x in [0, 1], t in [0, 1]
    u(x, 0) = sin(2*pi*x)   (initial condition)
    u(0, t) = u(1, t)       (periodic boundary condition)

three ways:

1. **Fortran90** (`fortran/`) — a traditional explicit finite-difference
   (first-order upwind) CFD solver.
2. **Python/NumPy** (`cfd/`) — the identical scheme, vectorized with NumPy,
   used to benchmark against the Fortran solver.
3. **PINN** (`pinn/`) — a small PyTorch MLP `u_theta(x, t)` trained so its
   automatic derivatives satisfy the PDE residual, initial condition, and
   periodic boundary condition. This is the neural-network approach; Python
   is used here because that's where the ML tooling (PyTorch/autograd) lives.

The exact solution `u(x, t) = sin(2*pi*(x - c*t))` is used to validate both
the CFD solvers (they should agree with each other and with this) and the
PINN (measured as relative L2 error), never during training.

## Project layout

```
fortran/
  advection.f90     Fortran90 CFD solver (explicit upwind, periodic BC)
cfd/
  advection.py       Equivalent Python/NumPy CFD solver
benchmark/
  compare.py         Times both solvers across grid sizes, checks they
                      agree, plots runtime and speedup
pinn/
  model.py           MLP network definition
  problem.py         exact solution + PDE/IC/BC residual losses
  train.py           training loop + CLI entry point
results/              benchmark CSV/plots + solver output (created on run)
outputs/               PINN plots and metrics (created on run)
```

## Setup

Requires a Fortran90 compiler (`gfortran`) and Python 3. Everything else is
handled by the Makefile:

```bash
make venv       # create .venv and install requirements.txt
make fortran    # compile bin/advection_fortran
```

## Compare Fortran90 vs. Python CFD runtime

```bash
make benchmark                          # default grid sizes: 500..8000
make benchmark SIZES="1000 4000 16000" TRIALS=5
```

This builds both solvers, times each across the given grid sizes (best of
`TRIALS` runs per size), confirms the two solutions agree to numerical
precision, and writes `results/benchmark.csv` and `results/benchmark.png`
(log-log runtime plot + Fortran speedup factor).

Run a single solver directly:

```bash
make run-fortran NX=8000
make run-python NX=8000
```

## Train the PINN

```bash
make pinn
```

Equivalent to `python -m pinn.train`. Useful flags: `--epochs`, `--lr`,
`--c` (advection speed), `--n-collocation`, `--seed`, `--outdir`. Run
`.venv/bin/python -m pinn.train --help` for the full list.

Training prints the PDE/IC/BC/total loss periodically and, on completion,
writes `outputs/loss_history.png` (training curves) and
`outputs/solution_comparison.png` (predicted vs. exact solution snapshots
plus the spatiotemporal error field) along with an `outputs/metrics.json`
summary (final losses and relative L2 error against the exact solution).

## Cleanup

```bash
make clean       # remove build/run artifacts (bin/, results/, outputs/)
make distclean    # clean + remove .venv
```
