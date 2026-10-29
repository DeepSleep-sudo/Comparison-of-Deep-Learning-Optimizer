"""
================================================================================
make_matrix.py  --  assemble the 3-task x 14-optimizer comparison matrix
================================================================================
Reads the three scenario logs and produces

    results/logs/matrix_summary.json          tidy numbers (single source of truth)
    results/figures/matrix_three_tasks.png    three-panel bar chart with error bars
    results/figures/matrix_rank_heatmap.png   per-task rank matrix
    latex/experiments/tables/table_matrix.tex LaTeX table for the paper

Every number printed by the report is read from matrix_summary.json, which is
produced here, which in turn is produced from the raw logs -- nothing is typed
by hand.

Definitions (fixed here so the report cannot drift from the code)
---------------------------------------------------------------
bench      mean over the five analytic landscapes of log10(final f(x)),
           skipping a landscape in which that optimizer diverged
           (final > 1e10 or non-finite).  Lower is better.
mnist      test accuracy in percent, mean over seeds; std over seeds.
transformer  mean of the last 50 training steps, cross-entropy; mean over seeds.

Run:
    python src/utils/make_matrix.py
================================================================================
"""
import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGS = os.path.join(ROOT, "results", "logs")
FIG = os.path.join(ROOT, "results", "figures")
TAB = os.path.join(ROOT, "latex", "experiments", "tables")
os.makedirs(FIG, exist_ok=True)

# The fourteen optimizers the report compares, in display order, with the
# family they belong to.  The family names are spelled out because a reader
# should not have to look up "F3" to understand a row.
OPTIMIZERS = [
    ("SGD",          "temporal filtering (F1)"),
    ("Momentum",     "temporal filtering (F1)"),
    ("Nesterov",     "temporal filtering (F1)"),
    ("Lion",         "temporal filtering (F1)"),
    ("AdaGrad",      "diagonal scaling (F2)"),
    ("RMSProp",      "diagonal scaling (F2)"),
    ("Adam",         "diagonal scaling (F2)"),
    ("AdamW",        "diagonal scaling (F2)"),
    ("Shampoo",      "matrix preconditioning (F3)"),
    ("Sophia",       "matrix preconditioning (F3)"),
    ("SOAP",         "matrix preconditioning (F3)"),
    ("Muon",         "spectral (F4)"),
    ("LAMB",         "meta-structure (F5)"),
    ("ScheduleFree", "meta-structure (F5)"),
]
DISPLAY = {"Nesterov": "NAG", "ScheduleFree": "Schedule-Free"}
DIVERGED = 1e10
# Floor for the synthetic-landscape objective.  Several optimizers drive the
# ill-conditioned quadratic to 1e-44 or to exact zero, and taking log10 of that
# measures how far float64 keeps descending rather than how well the optimizer
# works; a raw zero would even look infinitely bad.  Every value below 1e-16 is
# therefore reported as 1e-16: that is the level at which the objective of this
# problem is numerically indistinguishable from its minimum.  The floor is the
# same for every optimizer, so it cannot favour one of them.
FLOOR = 1e-16


def load(name):
    with open(os.path.join(LOGS, name), encoding="utf-8") as f:
        return json.load(f)


def bench_metric(bench):
    """mean log10(final) over the landscapes that did not diverge."""
    out = {}
    for name, _ in OPTIMIZERS:
        logs, n_div, per_fn = [], 0, {}
        for fname, tbl in bench.items():
            rec = tbl.get(name)
            if rec is None:
                continue
            v = rec["final"]
            if not math.isfinite(v) or v > DIVERGED:
                n_div += 1
                per_fn[fname] = None
                continue
            logs.append(math.log10(max(v, FLOOR)))
            per_fn[fname] = v
        out[name] = dict(log10_mean=float(np.mean(logs)) if logs else None,
                         n_diverged=n_div, per_function=per_fn)
    return out


def mnist_metric(m):
    out = {}
    for name, _ in OPTIMIZERS:
        rec = m.get(name)
        if rec is None:
            out[name] = dict(acc=None, std=None, n=None, lr=None)
            continue
        out[name] = dict(acc=100.0 * rec["acc"],
                         std=100.0 * rec.get("acc_std", 0.0),
                         n=rec.get("n_seeds", 1),
                         lr=rec["lr"])
    return out


def tf_metric(t):
    out = {}
    for name, _ in OPTIMIZERS:
        rec = t.get(name)
        if rec is None:
            out[name] = dict(loss=None, std=None, n=None, lr=None)
            continue
        out[name] = dict(loss=rec["final_loss"],
                         std=rec.get("final_loss_std", 0.0),
                         n=rec.get("n_seeds", 1),
                         lr=rec["lr"])
    return out


def ranks(values, higher_better):
    """1 = best; missing values get the worst rank + 1 and are marked."""
    idx = [i for i, v in enumerate(values) if v is not None]
    order = sorted(idx, key=lambda i: values[i], reverse=higher_better)
    r = [None] * len(values)
    for pos, i in enumerate(order, start=1):
        r[i] = pos
    return r


def plot_bars_paper(summ):
    """Vertical-bar variant sized for a 5.5-inch text column.

    The wide horizontal layout is fine on screen but shrinks to roughly 5pt type
    when scaled to \\linewidth, so the paper gets its own layout: three stacked
    panels on a shared x-axis, one colour per family.  Nothing else changes --
    same numbers, same metric definitions.
    """
    names = [DISPLAY.get(n, n) for n, _ in OPTIMIZERS]
    fams = [f for _, f in OPTIMIZERS]
    b = [summ["bench"][n]["log10_mean"] for n, _ in OPTIMIZERS]
    m = [summ["mnist"][n]["acc"] for n, _ in OPTIMIZERS]
    ms = [summ["mnist"][n]["std"] or 0.0 for n, _ in OPTIMIZERS]
    t = [summ["transformer"][n]["loss"] for n, _ in OPTIMIZERS]
    ts = [summ["transformer"][n]["std"] or 0.0 for n, _ in OPTIMIZERS]

    fam_colors = {}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for f in dict.fromkeys(fams):
        fam_colors[f] = palette[len(fam_colors) % len(palette)]
    colors = [fam_colors[f] for f in fams]

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.2), sharex=True)
    x = np.arange(len(names))
    specs = [
        (axes[0], b, None, 0.0,
         "(a) Synthetic landscapes: mean $\\log_{10}$ of the final objective (lower is better)",
         "$\\log_{10} f(x)$"),
        (axes[1], m, ms, 90.0,
         "(b) MNIST, small CNN: test accuracy (higher is better; bars start at 90\\%)",
         "accuracy (\\%)"),
        (axes[2], t, ts, 0.0,
         "(c) Small Transformer: final cross-entropy (lower is better)",
         "cross-entropy"),
    ]
    for ax, vals, errs, base, title, ylab in specs:
        h = [0.0 if v is None else v - base for v in vals]
        ax.bar(x, h, bottom=base, color=colors, width=0.74, alpha=0.9,
               yerr=errs if errs and any((e or 0) > 0 for e in errs) else None,
               error_kw=dict(ecolor="black", lw=0.8, capsize=2))
        ax.set_title(title, fontsize=8.6, loc="left")
        ax.set_ylabel(ylab, fontsize=8.2)
        ax.tick_params(axis="y", labelsize=7.6)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(names, rotation=40, ha="right", fontsize=7.8)

    short = {"temporal filtering (F1)": "F1 temporal filtering",
             "diagonal scaling (F2)": "F2 diagonal scaling",
             "matrix preconditioning (F3)": "F3 matrix preconditioning",
             "spectral (F4)": "F4 spectral",
             "meta-structure (F5)": "F5 meta-structure"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_colors.values()]
    fig.legend(handles, [short.get(k, k) for k in fam_colors], loc="lower center",
               ncol=3, fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"matrix_three_tasks_paper.{ext}"), dpi=300)
    plt.close(fig)
    print("[figure] matrix_three_tasks_paper.png / .pdf")


def plot_bars(summ):
    names = [DISPLAY.get(n, n) for n, _ in OPTIMIZERS]
    fams = [f for _, f in OPTIMIZERS]
    b = [summ["bench"][n]["log10_mean"] for n, _ in OPTIMIZERS]
    m = [summ["mnist"][n]["acc"] for n, _ in OPTIMIZERS]
    ms = [summ["mnist"][n]["std"] or 0.0 for n, _ in OPTIMIZERS]
    t = [summ["transformer"][n]["loss"] for n, _ in OPTIMIZERS]
    ts = [summ["transformer"][n]["std"] or 0.0 for n, _ in OPTIMIZERS]

    # one colour per family so the reader can see structure, not just order
    fam_colors = {}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for f in dict.fromkeys(fams):
        fam_colors[f] = palette[len(fam_colors) % len(palette)]
    colors = [fam_colors[f] for f in fams]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    y = np.arange(len(names))

    panels = [
        (axes[0], b, None, "(a) Synthetic landscapes\nmean $\\log_{10}$ final $f(x)$  (lower better)",
         "log10 final f(x)", 0.0),
        (axes[1], m, ms, "(b) MNIST, small CNN\ntest accuracy %  (higher better, axis starts at 90)",
         "accuracy (%)", 90.0),
        (axes[2], t, ts, "(c) Tiny Transformer LM\nfinal cross-entropy  (lower better)",
         "cross-entropy", 0.0),
    ]
    for ax, vals, errs, title, xlabel, left in panels:
        vals_plot = [left if v is None else v for v in vals]
        widths = [0.0 if v is None else v - left for v in vals]
        ax.barh(y, widths, left=left, color=colors, alpha=0.88,
                xerr=errs if errs and any((e or 0) > 0 for e in errs) else None,
                error_kw=dict(ecolor="black", lw=0.9, capsize=2.5))
        for i, v in enumerate(vals):
            if v is None:  # not evaluated: mark instead of drawing a zero bar
                ax.text(left, i, "  not run", va="center", fontsize=7, color="crimson")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # annotate the two extremes in each panel
    for ax, vals, better_low in ((axes[0], b, True), (axes[1], m, False), (axes[2], t, True)):
        ok = [(i, v) for i, v in enumerate(vals) if v is not None]
        bi = (min if better_low else max)(ok, key=lambda kv: kv[1])[0]
        wi = (max if better_low else min)(ok, key=lambda kv: kv[1])[0]
        for i, tag in ((bi, "best"), (wi, "worst")):
            v = vals[i]
            ax.annotate(tag, (v, i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, fontweight="bold", color="black")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_colors.values()]
    fig.legend(handles, list(fam_colors), loc="lower center", ncol=6,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Three-scenario comparison of fourteen optimizers "
                 "(error bars = std over 3 seeds; synthetic landscapes are deterministic)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"matrix_three_tasks.{ext}"), dpi=200)
    plt.close(fig)
    print("[figure] matrix_three_tasks.png / .pdf")


def plot_heatmap(summ):
    names = [DISPLAY.get(n, n) for n, _ in OPTIMIZERS]
    b = [summ["bench"][n]["log10_mean"] for n, _ in OPTIMIZERS]
    m = [summ["mnist"][n]["acc"] for n, _ in OPTIMIZERS]
    t = [summ["transformer"][n]["loss"] for n, _ in OPTIMIZERS]
    rb = ranks(b, higher_better=False)
    rm = ranks(m, higher_better=True)
    rt = ranks(t, higher_better=False)
    R = np.array([[np.nan if r is None else r for r in (rb[i], rm[i], rt[i])]
                  for i in range(len(names))], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 6.6))
    im = ax.imshow(R, cmap="RdYlGn_r", vmin=1, vmax=len(names), aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Synthetic\nlandscapes", "MNIST\n(CNN)", "Transformer\n(LM)"], fontsize=9)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(3):
            v = R[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=8.5, color="grey")
                continue
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8.5,
                    color="black" if 3 <= v <= len(names) - 3 else "white",
                    fontweight="bold" if v <= 3 else "normal")
    ax.set_title("Rank within each scenario (1 = best, 14 = worst)", fontsize=10.5)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("rank", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"matrix_rank_heatmap.{ext}"), dpi=200)
    plt.close(fig)
    print("[figure] matrix_rank_heatmap.png / .pdf")
    return dict(bench=rb, mnist=rm, transformer=rt)


def plot_heatmap_paper(summ):
    """Transposed, compact variant for the paper.

    Three scenario rows across fourteen optimizer columns instead of fourteen
    rows across three columns: the same numbers, but the figure is wide and short
    (about 2 inches tall at column width) instead of tall and narrow (about 5),
    which is what the page budget needs.  The column order matches the bar chart,
    so a method sits at the same horizontal position in both figures.
    """
    names = [DISPLAY.get(n, n) for n, _ in OPTIMIZERS]
    b = [summ["bench"][n]["log10_mean"] for n, _ in OPTIMIZERS]
    m = [summ["mnist"][n]["acc"] for n, _ in OPTIMIZERS]
    t = [summ["transformer"][n]["loss"] for n, _ in OPTIMIZERS]
    rb, rm, rt = ranks(b, False), ranks(m, True), ranks(t, False)
    R = np.array([[np.nan if v is None else v for v in (rb[i], rm[i], rt[i])]
                  for i in range(len(names))], dtype=float).T          # 3 x 14

    fig, ax = plt.subplots(figsize=(6.9, 2.45))
    im = ax.imshow(R, cmap="RdYlGn_r", vmin=1, vmax=len(names), aspect="auto")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Synthetic landscapes", "MNIST (CNN)", "Transformer (LM)"],
                       fontsize=8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7.6)
    for i in range(3):
        for j in range(len(names)):
            v = R[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=7.5, color="grey")
                continue
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7.6,
                    color="black" if 3 <= v <= len(names) - 3 else "white",
                    fontweight="bold" if v <= 3 else "normal")
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cb.set_label("rank", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"matrix_rank_heatmap_paper.{ext}"), dpi=300)
    plt.close(fig)
    print("[figure] matrix_rank_heatmap_paper.png / .pdf")


def write_table(summ, rk):
    if not os.path.isdir(os.path.join(ROOT, "latex")):
        print("[table] skipped: this checkout has no latex/ tree")
        return
    os.makedirs(TAB, exist_ok=True)
    rows = []
    for i, (name, fam) in enumerate(OPTIMIZERS):
        d = DISPLAY.get(name, name)
        b = summ["bench"][name]
        m = summ["mnist"][name]
        t = summ["transformer"][name]
        nd = b["n_diverged"]
        bcell = "--" if b["log10_mean"] is None else f"{b['log10_mean']:.2f}"
        if nd:
            bcell += f"$^{{\\dagger{nd}}}$"
        mcell = "--" if m["acc"] is None else f"{m['acc']:.2f}"
        mstd = "" if m["std"] is None else f" $\\pm$ {m['std']:.2f}"
        tcell = "--" if t["loss"] is None else f"{t['loss']:.3f}"
        tstd = "" if t["std"] is None else f" $\\pm$ {t['std']:.3f}"
        rr = ["--" if r is None else str(r)
              for r in (rk['bench'][i], rk['mnist'][i], rk['transformer'][i])]
        rows.append(f"    {d} & {fam} & {bcell} & {mcell}{mstd} & {tcell}{tstd} & "
                    f"{rr[0]}/{rr[1]}/{rr[2]} \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
  \caption{Three-scenario comparison of the fourteen optimizers. Synthetic:
  mean $\log_{10}$ of the final objective over the five analytic landscapes,
  skipping landscapes in which the run diverged ($\dagger k$ = $k$ divergences,
  a divergence is a final value above $10^{10}$). MNIST: test accuracy, mean
  $\pm$ std over three seeds. Transformer: cross-entropy averaged over the last
  50 training steps, mean $\pm$ std over three seeds. The last column is the
  rank in each scenario (synthetic / MNIST / transformer, 1 = best). Every
  optimizer is tuned over three learning rates centred on the value its own
  authors recommend, so the tuning budget is equal across families.}
  \label{tab:matrix}
  \centering
  \footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \begin{tabular}{@{}llcccc@{}}
    \toprule
    Optimizer & Family & Synthetic & MNIST acc.\ (\%) & Transformer loss & Rank \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    path = os.path.join(TAB, "table_matrix.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[table] {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", action="store_true",
                    help="also emit column-width variants of the figures for LaTeX")
    args = ap.parse_args()

    summ = dict(bench=bench_metric(load("bench_test_functions.json")),
                mnist=mnist_metric(load("mnist_results.json")),
                transformer=tf_metric(load("transformer_results.json")))
    rk = None
    out = {}
    for name, fam in OPTIMIZERS:
        out[name] = dict(family=fam, **{k: summ[k][name] for k in summ})
    path = os.path.join(LOGS, "matrix_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[json] {path}")

    plot_bars(summ)
    if args.paper:
        plot_bars_paper(summ)
    rk = plot_heatmap(summ)
    if args.paper:
        plot_heatmap_paper(summ)
    write_table(summ, rk)

    # console digest
    print("\n%-14s %-28s %9s %14s %16s" % ("optimizer", "family", "bench", "mnist", "transformer"))
    for name, fam in OPTIMIZERS:
        b, m, t = out[name]["bench"], out[name]["mnist"], out[name]["transformer"]
        print("%-14s %-28s %9s %14s %16s" % (
            DISPLAY.get(name, name), fam,
            "--" if b["log10_mean"] is None else f"{b['log10_mean']:.2f}",
            "--" if m["acc"] is None else f"{m['acc']:.2f}+-{m['std']:.2f}",
            "--" if t["loss"] is None else f"{t['loss']:.3f}+-{t['std']:.3f}"))


if __name__ == "__main__":
    main()
