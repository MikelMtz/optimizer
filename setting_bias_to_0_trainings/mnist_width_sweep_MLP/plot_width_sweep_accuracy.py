"""Width-Sweep Test Accuracy Plots
==================================

Produces a single combined figure (two rows):

  Row 1 — Line plots: mean test accuracy vs. network width
           (one line per initialisation scheme, panels for G&C and SGD)
  Row 2 — Heatmaps:  init × width grid coloured by mean test accuracy
           (panels for G&C and SGD)

Both rows share the same width axis (log₂ scale) and the same colour
palette so the two views can be read together.

Usage
-----
    python plot_width_sweep_accuracy.py \\
        --output_base output/width_sweep_mnist/output \\
        --save_path   output/width_sweep_mnist/plots/width_sweep_accuracy.pdf
"""

import argparse
import os
import sqlite3

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import numpy as np

# Default plotting labels' sizes, some changed manually for better aesthetics
matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
})

# ── Constants ─────────────────────────────────────────────────────────────────
ALL_WIDTHS = [16, 32, 64, 128, 256, 512]

INITS = ["uniform", "uniform_02", "kaiming_uniform", "kaiming_normal"]
INIT_LABELS = {
    "uniform":         r"$U[-1,1]$",
    "uniform_02":      r"$U[-0.2,0.2]$",
    "kaiming_uniform": "Kaiming Uniform",
    "kaiming_normal":  "Kaiming Normal",
}

OPTS = [("guess", "G&C"), ("sgd", "SGD")]

# Colour-blind-friendly palette (4 distinct colours)
INIT_COLORS = {
    "uniform":         "#0077BB",   # blue
    "uniform_02":      "#EE7733",   # orange
    "kaiming_uniform": "#009988",   # teal
    "kaiming_normal":  "#CC3311",   # red
}
INIT_MARKERS = {
    "uniform":         "o",
    "uniform_02":      "s",
    "kaiming_uniform": "^",
    "kaiming_normal":  "D",
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_db_stats(output_base, widths):
    """Return dict[opt_key][init][width] = (mean_acc, std_acc, n_runs).

    Reads *test_acc* directly from the SQLite databases — no model loading.
    """
    data = {opt: {init: {} for init in INITS} for opt, _ in OPTS}

    for opt_key, opt_label in OPTS:
        for init in INITS:
            for w in widths:
                tag = f"{opt_key}_{init}_u{w}"
                db_path = os.path.join(output_base, tag, "model_stats.db")
                if not os.path.exists(db_path):
                    continue
                try:
                    con = sqlite3.connect(db_path)
                    rows = con.execute(
                        "SELECT test_acc FROM model_stats WHERE status='COMPLETE'"
                    ).fetchall()
                    con.close()
                except Exception:
                    continue
                if not rows:
                    continue
                accs = np.array([r[0] for r in rows], dtype=float)
                data[opt_key][init][w] = (
                    float(accs.mean()),
                    float(accs.std()) if len(accs) > 1 else 0.0,
                    len(accs),
                )

    return data


def load_db_bins(output_base, widths):
    """Return per-row test_acc values for the scatter layer.

    dict[opt_key][init][width] = list of test_acc floats (one per completed row)

    NOTE: all runs in this sweep use the sentinel loss bin (-999, 999), so
    coloring by loss midpoint carries no information.  Instead we expose the
    individual per-seed test_acc values so the scatter shows run-to-run spread.
    """
    bins = {opt: {init: {} for init in INITS} for opt, _ in OPTS}

    for opt_key, _ in OPTS:
        for init in INITS:
            for w in widths:
                tag = f"{opt_key}_{init}_u{w}"
                db_path = os.path.join(output_base, tag, "model_stats.db")
                if not os.path.exists(db_path):
                    continue
                try:
                    con = sqlite3.connect(db_path)
                    rows = con.execute(
                        "SELECT test_acc FROM model_stats WHERE status='COMPLETE'"
                    ).fetchall()
                    con.close()
                except Exception:
                    continue
                accs = [r[0] for r in rows]
                if accs:
                    bins[opt_key][init][w] = accs

    return bins


# ── Helpers ───────────────────────────────────────────────────────────────────
def _width_ticks(widths):
    """Return uniformly spaced integer positions and string labels for the width axis."""
    positions = list(range(len(widths)))
    labels = [str(w) for w in widths]
    return positions, labels


def _global_acc_range(data, bins, widths, pad=0.003):
    """Return (y_min, y_max) spanning all per-seed and mean accuracies."""
    all_vals = []
    for opt_key, _ in OPTS:
        for init in INITS:
            for w in widths:
                if w in data[opt_key][init]:
                    all_vals.append(data[opt_key][init][w][0])
                if w in bins[opt_key][init]:
                    all_vals.extend(bins[opt_key][init][w])
    if not all_vals:
        return 0.95, 1.01
    return max(0.0, min(all_vals) - pad), min(1.0, max(all_vals) + pad)


# ── Row 1: Line plots ─────────────────────────────────────────────────────────
def plot_line_row(axes, data, bins, widths, y_min, y_max):
    """Draw one line-plot panel per optimizer into *axes* (length 2).

    Scatter layer: per-seed raw test_acc points (same colour as the mean line,
                   reduced alpha) — shows run-to-run spread directly.
    Line layer:    mean ± std across all seeds.
    """
    pos, w_labels = _width_ticks(widths)
    w_to_pos = {w: p for w, p in zip(widths, pos)}
    stride = max(1, len(widths) // 8)

    for col_i, (opt_key, opt_label) in enumerate(OPTS):
        ax = axes[col_i]
        has_data = False

        for init in INITS:
            # ── scatter: one point per seed/row ───────────────────────────
            for w in widths:
                if w not in bins[opt_key][init]:
                    continue
                has_data = True
                accs = bins[opt_key][init][w]
                ax.scatter(
                    [w_to_pos[w]] * len(accs), accs,
                    color=INIT_COLORS[init],
                    s=18, marker=INIT_MARKERS[init],
                    alpha=0.35, linewidths=0,
                    zorder=2,
                )

            # ── mean line + error bars ────────────────────────────────────
            xs, ys, yerr = [], [], []
            for w in widths:
                if w in data[opt_key][init]:
                    mean, std, n = data[opt_key][init][w]
                    xs.append(w_to_pos[w])
                    ys.append(mean)
                    yerr.append(std)

            if not xs:
                continue
            has_data = True

            xs = np.array(xs)
            ys = np.array(ys)
            yerr = np.array(yerr)

            color  = INIT_COLORS[init]
            marker = INIT_MARKERS[init]

            ax.errorbar(xs, ys, yerr=yerr,
                        color=color, marker=marker,
                        markersize=5, linewidth=1.6,
                        elinewidth=1.2, capsize=4, capthick=1.2,
                        zorder=4)

        ax.set_title(opt_label, fontweight="bold", pad=5, fontsize=15)
        ax.set_xlim(-0.5, len(widths) - 0.5)
        # Y: display up to 100.2% for breathing room; ticks stop at 100%
        y_display_max = 1.00
        ax.set_ylim(y_min, y_display_max)
        span = 1.0 - y_min
        tick_step = next(
            s for s in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
            if span / s <= 8
        )
        y_ticks = np.arange(
            np.ceil(y_min / tick_step) * tick_step, 1.0 + 1e-9, tick_step
        )
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
        ax.set_xticks(pos[::stride])
        ax.set_xticklabels(w_labels[::stride], rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Network width", fontsize=12)
        #if col_i == 0:
        ax.set_ylabel("Mean test accuracy", fontsize=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.grid(axis="x", linestyle=":",  linewidth=0.5, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single shared legend centred below both line-plot panels
    shared_handles = [
        mlines.Line2D([], [], color=INIT_COLORS[i], marker=INIT_MARKERS[i],
                      markersize=5, linewidth=1.6, label=INIT_LABELS[i])
        for i in INITS
    ]
    fig = axes[0].get_figure()
    fig.legend(handles=shared_handles,
               title='Initialisations',
               title_fontsize=12,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.54),
               ncol=len(shared_handles),
               framealpha=0.85, handlelength=1.8, borderpad=0.5,
               fontsize=9,
               frameon=False)




# ── Row 2: Heatmaps ───────────────────────────────────────────────────────────
def plot_heatmap_row(axes, data, widths, y_min, y_max, fig):
    """Draw one heatmap panel per optimizer into *axes* (length 2).

    Rows = init schemes (4), Columns = widths.
    Cell colour = mean test accuracy.  Grey = no data.
    """
    log2_w, w_labels = _width_ticks(widths)
    stride = max(1, len(widths) // 8)
    n_init = len(INITS)
    n_w    = len(widths)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#d0d0d0")

    images = []
    for col_i, (opt_key, opt_label) in enumerate(OPTS):
        ax = axes[col_i]

        grid = np.full((n_init, n_w), np.nan)
        for ri, init in enumerate(INITS):
            for ci, w in enumerate(widths):
                if w in data[opt_key][init]:
                    grid[ri, ci] = data[opt_key][init][w][0]   # mean acc

        im = ax.imshow(
            grid,
            aspect="auto",
            cmap=cmap,
            vmin=y_min, vmax=y_max,
            interpolation="nearest",
        )
        images.append(im)

        # Annotate cells
        for ri in range(n_init):
            for ci in range(n_w):
                val = grid[ri, ci]
                if not np.isnan(val):
                    # Pick text colour for contrast
                    normed = (val - y_min) / max(y_max - y_min, 1e-9)
                    txt_col = "white" if normed < 0.5 else "black"
                    ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                            fontsize=7.5, color=txt_col)
                else:
                    ax.text(ci, ri, "NaN", ha="center", va="center",
                            fontsize=7.5, color="black")

        ax.set_xticks(range(n_w))
        ax.set_xticklabels(w_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_init))
        ax.set_yticklabels([INIT_LABELS[i] for i in INITS], fontsize=9)
        ax.set_xlabel("Network width", fontsize=14)

        ax.set_title(opt_label, fontweight="bold", pad=5, fontsize=15)

    # Shared colourbar
    if images:
        cbar = fig.colorbar(
            images[0], ax=axes,
            orientation="vertical", fraction=0.035, pad=0.02,
            shrink=0.9,
        )
        cbar.set_label("Mean test accuracy", fontsize=12)
        cbar.ax.yaxis.set_major_formatter(
            ticker.PercentFormatter(xmax=1, decimals=1))
        cbar.ax.tick_params(labelsize=10)


# ── Summary printout ─────────────────────────────────────────────────────────
def print_line_summary(data, widths):
    """Print a table of mean ± std test accuracy for every (optimizer, init, width)."""
    col_w = max(len(lbl) for lbl in INIT_LABELS.values()) + 2
    width_header = "  ".join(f"{w:>6}" for w in widths)
    sep = "-" * (col_w + 2 + len(width_header))

    for opt_key, opt_label in OPTS:
        print(f"\n{'═' * len(sep)}")
        print(f"  {opt_label} — Mean test accuracy (mean ± std)")
        print(f"{'═' * len(sep)}")
        print(f"{'Init':<{col_w}}  {width_header}")
        print(sep)
        for init in INITS:
            row_parts = []
            for w in widths:
                if w in data[opt_key][init]:
                    mean, std, n = data[opt_key][init][w]
                    row_parts.append(f"{mean:.4f}±{std:.4f}" if std > 0 else f"{mean:.4f}      ")
                else:
                    row_parts.append("  —   ")
            print(f"{INIT_LABELS[init]:<{col_w}}  {'  '.join(f'{p:>13}' for p in row_parts)}")
        print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_base", default="output/width_sweep_mnist/output")
    p.add_argument("--widths", type=int, nargs="+", default=ALL_WIDTHS)
    p.add_argument("--save_path",
                   default="output/width_sweep_mnist/plots/width_sweep_accuracy.pdf")
    p.add_argument("--acc_ymin", type=float, default=None,
                   help="Fixed lower y-limit for accuracy (None = auto)")
    p.add_argument("--acc_ymax", type=float, default=None,
                   help="Fixed upper y-limit for accuracy (None = auto)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading data from SQLite databases …")
    data = load_db_stats(args.output_base, args.widths)
    bins = load_db_bins(args.output_base, args.widths)

    y_min, y_max = _global_acc_range(data, bins, args.widths)
    if args.acc_ymin is not None:
        y_min = args.acc_ymin
    if args.acc_ymax is not None:
        y_max = args.acc_ymax

    print(f"Accuracy range for plots:   [{y_min:.4f}, {y_max:.4f}]")

    print_line_summary(data, args.widths)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10), constrained_layout=False)
    gs = mgridspec.GridSpec(
        2, 2, figure=fig,
        left=0.08, right=0.97,
        top=0.88, bottom=0.08,
        hspace=0.4, wspace=0.3,
        height_ratios=[1.5, 3],
    )

    # Row 1: line plots
    ax_line_gc  = fig.add_subplot(gs[0, 0])
    ax_line_sgd = fig.add_subplot(gs[0, 1])
    # Row 2: heatmaps (square-like: ~4.9" tall × ~4.6" wide per panel)
    ax_heat_gc  = fig.add_subplot(gs[1, 0])
    ax_heat_sgd = fig.add_subplot(gs[1, 1])

    plot_line_row([ax_line_gc, ax_line_sgd], data, bins,
                  args.widths, y_min, y_max)

    plot_heatmap_row(
        [ax_heat_gc, ax_heat_sgd], data, args.widths, y_min, y_max, fig)

    fig.suptitle(
        "mnist Dataset — Test Accuracy vs. Network Width\n"
        "G&C (Guess-and-Check) vs. SGD  |  All Initialisation Schemes",
        fontsize=12, fontweight="bold", y=0.95,
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    plt.savefig(args.save_path, bbox_inches="tight", dpi=150)
    print(f"Figure saved → {args.save_path}")
    plt.close()


if __name__ == "__main__":
    main()
