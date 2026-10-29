"""
================================================================================
test_functions.py  --  Optimizer benchmark on classic optimisation landscapes
================================================================================
Before training neural nets, we test every optimizer on smooth analytic
landscapes where the geometry is known. This isolates *optimizer behaviour*
from *model behaviour*.

Functions
---------
    Quadratic (ill-conditioned)  : f = 0.5 x^T Q x, Q with condition number kappa
    Rosenbrock                   : narrow curved valley (the "banana")
    Rastrigin                    : highly multimodal (many local minima)
    Beale                        : flat plateau + narrow curved valley
    Ackley                       : nearly flat outer region + central hole

Protocol
--------
    * All optimizers start from the same x0.
    * For fairness we sweep a small grid of learning rates per optimizer and
      report the *best* (lowest final loss) run -> "best-tuned" comparison.
    * Loss is evaluated on the *deterministic* function (no noise), so this
      measures pure optimisation ability.

Run:
    python src/experiments/test_functions.py --device cuda:1 --steps 3000
================================================================================
"""
import os
import sys
import json
import argparse
import itertools

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.optimizers import OPTIMIZERS
from src.utils.plotting import plot_loss_curves, save_json, set_style
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Loss landscapes (all operate on a torch tensor x of shape [D])
# ----------------------------------------------------------------------------
def make_quadratic(D=10, kappa=100.0):
    """f(x) = 0.5 x^T Q x with Q = diag(logspace(0, log10(kappa), D)).
    Hessian eigenvalues span [1, kappa] -> condition number = kappa."""
    def f(x):
        eig = torch.logspace(0, np.log10(kappa), D, device=x.device, dtype=x.dtype)
        return 0.5 * (eig * x ** 2).sum()
    return f, f"Quadratic (kappa={kappa:g})"


def rosenbrock(x):
    """f = sum_{i} [ (1-x_i)^2 + 100 (x_{i+1} - x_i^2)^2 ]. Global min 0 at x=1."""
    return ((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2).sum()


def rastrigin(x):
    """f = 10D + sum [ x_i^2 - 10 cos(2 pi x_i) ]. Global min 0 at x=0."""
    D = x.numel()
    return 10 * D + (x ** 2 - 10 * torch.cos(2 * np.pi * x)).sum()


def beale(x):
    """Beale function (2-D). Global min 0 at (3, 0.5)."""
    x1, x2 = x[0], x[1]
    return ((1.5 - x1 + x1 * x2) ** 2
            + (2.25 - x1 + x1 * x2 ** 2) ** 2
            + (2.625 - x1 + x1 * x2 ** 3) ** 2)


def ackley(x):
    """Ackley function. Global min 0 at x=0."""
    D = x.numel()
    a, b, c = 20.0, 0.2, 2 * np.pi
    s1 = (x ** 2).sum().sqrt()
    s2 = torch.cos(c * x).sum()
    return -a * torch.exp(-b * s1 / (D ** 0.5)) \
           - torch.exp(s2 / D) + a + np.e


FUNCTIONS = {
    "Quadratic": dict(fn=None, dim=10, x0=2.0, steps=2000),
    "Rosenbrock": dict(fn=rosenbrock, dim=8, x0=-1.5, steps=4000),
    "Rastrigin": dict(fn=rastrigin, dim=8, x0=3.0, steps=4000),
    "Beale": dict(fn=beale, dim=2, x0=-2.0, steps=4000),
    "Ackley": dict(fn=ackley, dim=8, x0=3.0, steps=4000),
}


def run_one(opt_name, opt_cls, opt_kwargs, fn, x0, steps, lr, device):
    """Run a single optimizer on a single function; return loss history."""
    x = x0.clone().to(device).requires_grad_(True)
    kwargs = dict(opt_kwargs)
    kwargs["lr"] = lr
    # some optimizers need eps adjustments on these scales
    if opt_name in ("AdaBelief",):
        kwargs["eps"] = 1e-12
    try:
        opt = opt_cls([x], **kwargs)
    except TypeError:
        return None
    history = []
    for t in range(steps):
        opt.zero_grad()
        loss = fn(x)
        loss.backward()
        opt.step()
        with torch.no_grad():
            history.append(float(fn(x).detach().cpu()))
        if history[-1] > 1e12 or np.isnan(history[-1]):
            # diverged: pad and stop
            history.extend([history[-1]] * (steps - len(history)))
            break
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr_grid", type=float, nargs="+",
                    default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    ap.add_argument("--functions", nargs="+", default=None,
                    help="subset of function names (default: all)")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[test_functions] device = {device}")

    all_results = {}
    for fname, cfg in FUNCTIONS.items():
        if args.functions and fname not in args.functions:
            continue
        steps = args.steps or cfg["steps"]
        if fname == "Quadratic":
            fn, title = make_quadratic(cfg["dim"])
        else:
            fn = cfg["fn"]
            title = fname
        x0 = torch.full((cfg["dim"],), cfg["x0"], dtype=torch.float64)

        best_histories = {}
        best_info = {}
        for opt_name, (opt_cls, opt_kwargs) in OPTIMIZERS.items():
            if opt_name == "Lookahead":
                continue
            best = None
            for lr in args.lr_grid:
                h = run_one(opt_name, opt_cls, opt_kwargs, fn, x0, steps, lr, device)
                if h is None:
                    continue
                final = h[-1]
                if best is None or final < best[0]:
                    best = (final, lr, h)
            if best is not None:
                best_histories[opt_name] = best[2]
                best_info[opt_name] = dict(final=best[0], lr=best[1])
                print(f"  {fname:12s} {opt_name:14s} best_lr={best[1]:.1e} final={best[0]:.3e}")
        all_results[fname] = best_info
        plot_loss_curves(best_histories, f"{title}  (best lr per optimizer)",
                         f"bench_{fname.lower()}", ylabel="f(x)", logy=True)

    save_json(all_results, "bench_test_functions")
    print("done.")


if __name__ == "__main__":
    main()
