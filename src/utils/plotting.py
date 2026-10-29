"""
Shared plotting helpers for the optimizer experiments.

All figures are saved as both PNG (for reports) and PDF (for LaTeX) into
``results/figures``. A consistent, colour-blind-friendly palette is used.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "figures")
FIG_DIR = os.path.abspath(FIG_DIR)
os.makedirs(FIG_DIR, exist_ok=True)

# A 26-colour qualitative palette (tab20 + tab20b-ish), readable on light bg.
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5", "#393b79", "#637939", "#8c6d31", "#843c39",
    "#7b4173", "#5254a3",
]


def color_for(i):
    return PALETTE[i % len(PALETTE)]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def savefig(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    png = os.path.join(FIG_DIR, f"{name}.png")
    pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[figure] saved {png}")
    return png


def plot_loss_curves(histories, title, fname, ylabel="Loss", logy=True,
                     xlabel="Step", smooth=0):
    """histories: dict name -> list of loss values (or {'loss':..., 'acc':...})."""
    set_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (name, hist) in enumerate(histories.items()):
        y = hist["loss"] if isinstance(hist, dict) else hist
        y = np.asarray(y, dtype=float)
        if smooth and smooth > 1 and len(y) >= smooth:
            kernel = np.ones(smooth) / smooth
            y = np.convolve(y, kernel, mode="valid")
        ax.plot(y, label=name, color=color_for(i), lw=1.8)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    return savefig(fig, fname)


def save_json(obj, name):
    path = os.path.join(FIG_DIR, "..", "logs", f"{name}.json")
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[json] saved {path}")
    return path
