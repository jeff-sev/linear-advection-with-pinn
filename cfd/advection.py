"""1D linear advection, explicit first-order upwind scheme, periodic BC.

Same equation/IC/scheme as fortran/advection.f90, so runtimes and final
fields can be compared directly.

Usage: python cfd/advection.py [--nx NX]
"""
import argparse
import math
import time

import numpy as np

C = 1.0
CFL = 0.8
T_FINAL = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=4000)
    parser.add_argument("--out", type=str, default="results/python_output.csv")
    args = parser.parse_args()

    dx = 1.0 / args.nx
    dt = CFL * dx / C
    nt = math.ceil(T_FINAL / dt)

    x = np.arange(args.nx) * dx
    u = np.sin(2 * np.pi * x)

    start = time.perf_counter()
    for _ in range(nt):
        u = u - C * dt / dx * (u - np.roll(u, 1))
    elapsed = time.perf_counter() - start

    print(f"nx={args.nx}")
    print(f"nt={nt}")
    print(f"elapsed_seconds={elapsed:.6f}")

    np.savetxt(args.out, np.column_stack([x, u]), delimiter=",", header="x,u", comments="")


if __name__ == "__main__":
    main()
