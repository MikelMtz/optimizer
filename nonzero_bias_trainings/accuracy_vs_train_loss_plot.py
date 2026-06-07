"""Plot: Test Accuracy vs. Normalized Train Loss
================================================

Produces two figures:

1. **Heatmap grid** (saved to ``--save_path``):
   A grid of 2D density heatmap subplots.

   Rows (4):
     0  G&C  — Test accuracy  vs  weight (Frobenius) norm. loss
     1  G&C  — Test accuracy  vs  Lipschitz norm. loss
     2  SGD  — Test accuracy  vs  weight (Frobenius) norm. loss
     3  SGD  — Test accuracy  vs  Lipschitz norm. loss

   Columns: one per selected hidden-unit width.

2. **Scatter overlay** (saved to ``--scatter_save_path``):
   4 subplots (one per optimizer × normalisation combination).  Each
   subplot overlays the mean test-accuracy-per-loss-bin lines for all
   selected widths, colour-coded by width.

Usage (run from the check_bias_0/ directory):

    python accuracy_vs_train_loss_plot.py \\
        --output_base frobenius_norm/output \\
        --widths 4 16 64 256 \\
        --init kaiming_uniform \\
        --save_path plots/lipschitz_heatmap.pdf \\
        --scatter_save_path plots/lipschitz_scatter.pdf
"""
import os
import sys
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import MLPModels, calculate_loss_acc
from datasets import Kink

# ── Constants ─────────────────────────────────────────────────────────────────
ALL_WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
INITS = ["uniform", "uniform_02", "kaiming_uniform", "kaiming_normal"]

INIT_DISPLAY = {
    "uniform":          r"$U[-1,1]$",
    "uniform_02":       r"$U[-0.2,0.2]$",
    "kaiming_uniform":  "Kaiming Uniform",
    "kaiming_normal":   "Kaiming Normal",
}

# Row definitions: (optimizer_key, norm_mode, row_label_left, colormap)
ROWS = [
    ("guess", "frobenius",
     "G&C\nTest accuracy vs\nweight norm. loss",   "Blues"),
    ("guess", "lipschitz",
     "G&C\nTest accuracy vs\nLipschitz norm. loss", "Blues"),
    ("SGD",   "frobenius",
     "SGD\nTest accuracy vs\nweight norm. loss",    "Reds"),
    ("SGD",   "lipschitz",
     "SGD\nTest accuracy vs\nLipschitz norm. loss",  "Reds"),
]


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot test accuracy vs normalised train loss (2D density heatmaps)")
    p.add_argument("--output_base", default="frobenius_norm/output",
                   help="Base folder containing per-run directories")
    p.add_argument("--widths", type=int, nargs="+", default=[4, 16, 64, 256],
                   help="Hidden-unit widths to plot as columns")
    p.add_argument("--init", default="kaiming_uniform", choices=INITS,
                   help="Weight initialisation to display")
    p.add_argument("--test_samples", type=int, default=100,
                   help="Number of total samples for Kink test set (70/30 split)")
    p.add_argument("--test_seed", type=int, default=0)
    p.add_argument("--train_samples", type=int, default=16,
                   help="Number of total samples for Kink train set (70/30 split)")
    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--loss_bins", type=int, default=30,
                   help="Number of bins along the train-loss axis")
    p.add_argument("--acc_bins", type=int, default=40,
                   help="Number of bins along the test-accuracy axis")
    p.add_argument("--bin_count", type=int, default=15,
                   help="Number of bins for the per-bin mean-accuracy line")
    p.add_argument("--max_models", type=int, default=0,
                   help="Max models to use per experiment (0 = all). "
                        "Useful to limit memory for very wide networks.")
    p.add_argument("--save_path", default="plots/lipschitz_heatmap.pdf")
    p.add_argument("--scatter_save_path",
                   default="plots/lipschitz_scatter.pdf",
                   help="Save path for the width-colored scatter overlay figure")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_models(model_dir, device, max_models=0):
    """Load all model checkpoints from *model_dir* and return a list of
    MLPModels containers.  If *max_models* > 0, subsample each checkpoint
    to keep total model count manageable."""
    if not os.path.isdir(model_dir):
        return []
    files = sorted(glob.glob(os.path.join(model_dir, "*")))
    results = []
    total_loaded = 0
    for f in files:
        try:
            ckpt = torch.load(f, map_location=device)
        except Exception:
            continue
        kwargs = ckpt["kwargs"]
        sd = ckpt["good_models_state_dict"]
        mc = kwargs["model_count"]

        # Subsample if needed
        if max_models > 0 and total_loaded + mc > max_models:
            keep = max(max_models - total_loaded, 0)
            if keep == 0:
                break
            kwargs = dict(kwargs, model_count=keep)
            sd = {k: v[:, :keep] for k, v in sd.items()}
            mc = keep

        models = MLPModels(**kwargs, device=torch.device(device))
        models.load_state_dict(sd)
        results.append(models)
        total_loaded += mc
        if max_models > 0 and total_loaded >= max_models:
            break
    return results


def compute_metrics(models, train_data, train_labels, test_data, test_labels,
                    norm_mode):
    """Return (train_losses, test_accs) — each shape ``(model_count,)``.

    For Lipschitz mode the normalisation constant is computed **once** and
    reused, which is orders of magnitude faster than recomputing it for
    every data-sample batch inside ``forward_normalize_lipschitz``.
    """
    loss_func = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        if norm_mode == "lipschitz":
            ref_data = torch.cat([train_data, test_data], dim=0)
            lip_const = models.compute_lipschitz_constant(ref_data).clamp(min=1e-8)

            def fwd_fn(x):
                return models.forward(x) / lip_const[None, :, None]
        else:
            fwd_fn = models.forward_normalize_frobenius

        train_losses, _ = calculate_loss_acc(
            train_data, train_labels, fwd_fn, loss_func, batch_size=1,
        )
        _, test_accs = calculate_loss_acc(
            test_data, test_labels, fwd_fn, loss_func, batch_size=1,
        )
    return train_losses.cpu(), test_accs.cpu()


def bin_mean_acc(train_losses, test_accs, bin_count):
    """Mean test accuracy per train-loss bin.  Returns (centers, means)."""
    lo, hi = train_losses.min().item(), train_losses.max().item()
    if lo == hi:
        return [lo], [test_accs.mean().item()]
    edges = np.linspace(lo, hi, bin_count + 1)
    centers, means = [], []
    for l, u in zip(edges[:-1], edges[1:]):
        mask = (train_losses >= l) & (train_losses <= u)
        if mask.sum() > 0:
            centers.append((l + u) / 2)
            means.append(test_accs[mask].mean().item())
    return centers, means


# ── Data collection ───────────────────────────────────────────────────────────
def collect_all_metrics(args, device, train_data, train_labels,
                        test_data, test_labels):
    """Return a dict  ``metrics[(row_i, width)] = (tl_np, ta_np)``."""
    metrics = {}  # (row_i, width) → (tl_np, ta_np)

    for col_i, w in enumerate(args.widths):
        for row_i, (opt_key, norm_mode, row_label, _cmap) in enumerate(ROWS):
            tag = f"{opt_key.lower()}_{args.init}_u{w}"
            short_label = row_label.split("\n")[0]

            model_dir = os.path.join(args.output_base, tag, "models")
            print(f"[{short_label} / {norm_mode} / width={w}] "
                  f"Loading from {model_dir}")

            model_list = load_models(model_dir, args.device, args.max_models)
            if not model_list:
                continue

            all_tl, all_ta = [], []
            for models in model_list:
                try:
                    tl, ta = compute_metrics(
                        models, train_data, train_labels,
                        test_data, test_labels, norm_mode,
                    )
                    all_tl.append(tl)
                    all_ta.append(ta)
                except Exception as e:
                    print(f"  [warn] skipping: {e}")

            if not all_tl:
                continue

            tl_cat = torch.cat(all_tl)
            ta_cat = torch.cat(all_ta)
            print(f"  models: {len(tl_cat)}, "
                  f"mean test acc: {ta_cat.numpy().mean():.4f}")

            metrics[(row_i, w)] = (tl_cat.numpy(), ta_cat.numpy())

    return metrics


# ── Figure 1: Heatmap grid ───────────────────────────────────────────────────
def plot_heatmap(args, metrics):
    """Original 2-D density heatmap grid (rows × width columns)."""
    n_rows = len(ROWS)
    n_cols = len(args.widths)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols + 1.5, 3.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    row_mesh = [None] * n_rows

    for col_i, w in enumerate(args.widths):
        for row_i, (opt_key, norm_mode, row_label, cmap_name) in enumerate(ROWS):
            ax = axes[row_i, col_i]

            if (row_i, w) not in metrics:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                _decorate_ax(ax, row_i, col_i, n_rows, w, row_label)
                continue

            tl_np, ta_np = metrics[(row_i, w)]
            total = len(tl_np)
            mean_acc = ta_np.mean()

            # ── 2D histogram ────────────────────────────────────────────
            x_lo = max(tl_np.min() * 0.95, 0.0)
            x_hi = max(tl_np.max() * 1.05, 0.01)
            y_lo, y_hi = max(ta_np.min() - 0.05, 0.0), 1.02

            H, xedges, yedges = np.histogram2d(
                tl_np, ta_np,
                bins=[args.loss_bins, args.acc_bins],
                range=[[x_lo, x_hi], [y_lo, y_hi]],
            )
            H_pct = H / total * 100.0
            H_masked = np.ma.masked_where(H_pct == 0, H_pct)

            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad(color="white")

            pos_vals = H_pct[H_pct > 0]
            vmin = pos_vals.min() if len(pos_vals) else 0.01
            vmax = pos_vals.max() if len(pos_vals) else 1.0
            vmin = max(vmin * 0.5, 1e-3)

            im = ax.pcolormesh(
                xedges, yedges, H_masked.T,
                cmap=cmap,
                norm=LogNorm(vmin=vmin, vmax=vmax),
                rasterized=True,
            )
            if row_mesh[row_i] is None:
                row_mesh[row_i] = im

            # ── Bin-mean black dots ─────────────────────────────────────
            centers, means = bin_mean_acc(
                torch.tensor(tl_np), torch.tensor(ta_np), args.bin_count,
            )
            ax.scatter(centers, means, s=20, color="black", zorder=5,
                       edgecolors="black", linewidths=0.3)
            ax.plot(centers, means, color="black", linewidth=0.8, zorder=4)

            # ── Overall mean dashed line ────────────────────────────────
            ax.axhline(y=mean_acc,
                       color="red" if "SGD" in row_label else "blue",
                       linestyle="--", linewidth=0.8, alpha=0.6, zorder=3)

            # ── Legend box ──────────────────────────────────────────────
            ax.scatter([], [], s=20, color="black",
                       label="Mean test accuracy (per loss bin)")
            ax.plot([], [], "--",
                    color="red" if "SGD" in row_label else "blue",
                    linewidth=0.8, alpha=0.6,
                    label=f"Test accuracy mean (total) {mean_acc:.3f}")
            ax.legend(fontsize=5, loc="lower left", framealpha=0.85,
                      handlelength=1.5, borderpad=0.3, labelspacing=0.3)

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            _decorate_ax(ax, row_i, col_i, n_rows, w, row_label)

    for row_i in range(n_rows):
        if row_mesh[row_i] is not None:
            cbar = fig.colorbar(
                row_mesh[row_i],
                ax=axes[row_i, :].tolist(),
                pad=0.015, fraction=0.025, aspect=18,
            )
            cbar.set_label("Percentage (log scale)", fontsize=8)

    fig.suptitle(
        "Test Accuracy vs. Normalised Train Loss — Kink dataset\n"
        f"Init: {INIT_DISPLAY.get(args.init, args.init)}",
        fontsize=13, y=1.02,
    )
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    plt.savefig(args.save_path, bbox_inches="tight", dpi=150)
    print(f"\nHeatmap figure saved to: {args.save_path}")
    plt.close()


# ── Figure 2: Scatter overlay (width colour-coded) ──────────────────────────
def plot_scatter_overlay(args, metrics):
    """4 subplots (one per ROWS entry).  Each overlays per-bin mean-accuracy
    lines from every requested width, colour-coded by width."""
    n_rows = len(ROWS)

    # Build a colour map for the widths
    cmap = plt.get_cmap("tab10")
    width_colors = {w: cmap(i % 10) for i, w in enumerate(args.widths)}

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10, 7),
        squeeze=False,
        constrained_layout=True,
    )

    for row_i, (opt_key, norm_mode, row_label, _cmap_name) in enumerate(ROWS):
        ax = axes[row_i // 2, row_i % 2]
        has_data = False

        for w in args.widths:
            if (row_i, w) not in metrics:
                continue

            tl_np, ta_np = metrics[(row_i, w)]
            centers, means = bin_mean_acc(
                torch.tensor(tl_np), torch.tensor(ta_np), args.bin_count,
            )
            if not centers:
                continue

            color = width_colors[w]
            ax.scatter(centers, means, s=28, color=color, zorder=5,
                       edgecolors="black", linewidths=0.3)
            ax.plot(centers, means, color=color, linewidth=1.2, zorder=4,
                    label=f"Width {w}")
            has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")

        ax.set_ylabel("Test accuracy", fontsize=9)
        ax.set_xlabel("Train loss (normalized)", fontsize=9)
        # Multi-line row label as title
        short = row_label.replace("\n", " — ")
        ax.set_title(short, fontsize=10, fontweight="bold")
        if has_data:
            ax.legend(fontsize=7, loc="lower left", framealpha=0.85,
                      ncol=min(len(args.widths), 4))

    fig.suptitle(
        f"Mean Test Accuracy per Loss Bin for Kink dataset - Init: {INIT_DISPLAY.get(args.init, args.init)}",
        fontsize=13, y=1.02,
    )
    os.makedirs(os.path.dirname(args.scatter_save_path) or ".", exist_ok=True)
    plt.savefig(args.scatter_save_path, bbox_inches="tight", dpi=150)
    print(f"Scatter figure saved to: {args.scatter_save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- Evaluation datasets ------------------------------------------------
    train_ds = Kink(train=True,  samples=args.train_samples,
                    seed=args.train_seed, noise=0.0, margin=0.25)
    test_ds  = Kink(train=False, samples=args.test_samples,
                    seed=args.test_seed,  noise=0.0, margin=0.25)

    train_data   = torch.tensor(train_ds.data).float().to(device)
    train_labels = torch.tensor(train_ds.labels).long().to(device)
    test_data    = torch.tensor(test_ds.data).float().to(device)
    test_labels  = torch.tensor(test_ds.labels).long().to(device)

    # ---- Collect (or load cached) metrics -----------------------------------
    metrics = collect_all_metrics(args, device, train_data, train_labels,
                                  test_data, test_labels)

    # ---- Generate both figures ----------------------------------------------
    plot_heatmap(args, metrics)
    plot_scatter_overlay(args, metrics)


def _decorate_ax(ax, row_i, col_i, n_rows, width, row_label):
    """Apply shared axis labels and titles to *ax*."""
    if row_i == n_rows - 1:
        ax.set_xlabel("Train loss (normalized)", fontsize=9)
    if col_i == 0:
        ax.set_ylabel(row_label, fontsize=9)
    if row_i == 0:
        ax.set_title(f"Width {width}", fontsize=11, fontweight="bold")


if __name__ == "__main__":
    main()
