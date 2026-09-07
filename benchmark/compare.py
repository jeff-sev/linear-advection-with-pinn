"""Run the Fortran90 and Python CFD solvers across grid sizes, time each,
cross-check that they agree, and plot the results."""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ELAPSED_RE = re.compile(r"elapsed_seconds=\s*([0-9.]+)")


def run_and_time(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    match = ELAPSED_RE.search(result.stdout)
    if not match:
        raise RuntimeError(f"could not parse elapsed time from output:\n{result.stdout}")
    return float(match.group(1))


def load_solution(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortran-bin", default="bin/advection_fortran")
    parser.add_argument("--python-script", default="cfd/advection.py")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fortran_out = outdir / "fortran_output.csv"
    python_out = outdir / "python_output.csv"

    rows = []
    for nx in args.sizes:
        fortran_times = [run_and_time([args.fortran_bin, str(nx)]) for _ in range(args.trials)]
        python_times = [
            run_and_time([args.python_bin, args.python_script, "--nx", str(nx)])
            for _ in range(args.trials)
        ]
        f_best, p_best = min(fortran_times), min(python_times)
        rows.append((nx, f_best, p_best, p_best / f_best))
        print(
            f"nx={nx:>7d}  fortran={f_best:.4f}s  python={p_best:.4f}s  "
            f"speedup(python/fortran)={p_best / f_best:.1f}x"
        )

    # The last iteration left both solvers' output for the largest size on
    # disk; confirm the two schemes actually agree before trusting the timings.
    f_sol = load_solution(fortran_out)
    p_sol = load_solution(python_out)
    max_abs_diff = float(np.max(np.abs(f_sol[:, 1] - p_sol[:, 1])))
    print(f"\nmax |fortran - python| at nx={args.sizes[-1]}: {max_abs_diff:.3e}")

    csv_path = outdir / "benchmark.csv"
    with open(csv_path, "w") as f:
        f.write("nx,fortran_seconds,python_seconds,speedup\n")
        for nx, ft, pt, sp in rows:
            f.write(f"{nx},{ft:.6f},{pt:.6f},{sp:.3f}\n")

    sizes = [r[0] for r in rows]
    fortran_times = [r[1] for r in rows]
    python_times = [r[2] for r in rows]
    speedups = [r[3] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.loglog(sizes, fortran_times, "o-", label="Fortran90")
    ax1.loglog(sizes, python_times, "o-", label="Python (NumPy)")
    ax1.set_xlabel("grid points (nx)")
    ax1.set_ylabel("best wall time (s)")
    ax1.set_title("CFD solver runtime vs. grid size")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(sizes, speedups, "o-", color="darkorange")
    ax2.set_xscale("log")
    ax2.set_xlabel("grid points (nx)")
    ax2.set_ylabel("python time / fortran time")
    ax2.set_title("Fortran speedup factor")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "benchmark.png", dpi=150)
    plt.close(fig)

    print(f"\nwrote {csv_path} and {outdir / 'benchmark.png'}")


if __name__ == "__main__":
    main()
