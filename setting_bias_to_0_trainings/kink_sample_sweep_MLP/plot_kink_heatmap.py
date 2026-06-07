"""
plot_kink_heatmap.py
====================
Heatmap of test error (= 1 − test accuracy) as a function of
hidden width (or network depth) and number of training samples.

Layout: 4 rows × 2 columns
  rows    → initialisation  (uniform, uniform_02, kaiming_uniform, kaiming_normal)
  columns → optimiser       (G&C, SGD)
  x-axis  → sweep value     (width on log scale; depth on linear scale)
  y-axis  → # training samples
  colour  → test error

Bin edges for pcolormesh are computed as geometric midpoints (width) or
arithmetic midpoints (depth/samples) so each data point sits at the visual
centre of its cell.

Produces two figures:
  - kink_width_heatmap.png
  - kink_depth_heatmap.png

Usage (run from check_bias_0/):
    python kink_sample_sweep_MLP/plot_kink_heatmap.py

Options:
    --output_base      Parent of width/ and depth/ folders
                       [default: <script_dir>/output]
    --output_folder    Where to save plots
                       [default: <script_dir>/heatmaps]
    --widths           Hidden-unit widths  [default: 2 8 32 128 512]
    --depths           Network depths      [default: 1 2 3 4 5 6]
    --num_samples      Sample counts       [default: 2 4 6 8 12 16 20 26 30]
    --width_plot_name  [default: kink_width_heatmap]
    --depth_plot_name  [default: kink_depth_heatmap]
    --vmax             Colormap ceiling for test error [default: auto]
    --test_samples     Total samples for the fixed test set [default: 100]
    --test_seed        Seed for the fixed test set [default: 0]
    --device           torch device [default: cpu]
"""

import os
import sys
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from datasets import Kink
from utils_bias_0 import MLPModels, calculate_loss_acc

# ── Constants ──────────────────────────────────────────────────────────────────
INITS      = ["uniform", "uniform_02", "kaiming_uniform", "kaiming_normal"]
WIDTHS     = [2, 8, 32, 128, 512]
DEPTHS     = [1, 2, 3, 4, 5, 6]
NUM_SAMPLES_LIST = [2, 4, 6, 8, 12, 16, 20, 26, 30]
OPTIMISERS = ["guess", "sgd"]

INIT_LABELS = {
    "uniform":         "Uniform [-1,1]",
    "uniform_02":      "Uniform [-0.2,0.2]",
    "kaiming_uniform": "Kaiming Uniform",
    "kaiming_normal":  "Kaiming Normal",
}
OPT_LABELS = {"guess": "G&C", "sgd": "SGD"}

KINK_MARGIN = 0.25
KINK_NOISE  = 0.0

CMAP = "inferno"


# ── Data helpers ───────────────────────────────────────────────────────────────

def get_test_data(samples: int, seed: int, device: torch.device):
    kink = Kink(train=False, samples=samples, seed=seed,
                noise=KINK_NOISE, margin=KINK_MARGIN)
    test_data   = torch.tensor(kink.data,   dtype=torch.float32).to(device)
    test_labels = torch.tensor(kink.labels, dtype=torch.long).to(device)
    return test_data, test_labels


def load_checkpoint_models(path: str, device: torch.device) -> MLPModels:
    ckpt   = torch.load(path, map_location=device)
    kwargs = dict(ckpt["kwargs"])
    kwargs["device"] = device
    models = MLPModels(**kwargs)
    models.load_state_dict(ckpt["good_models_state_dict"])
    return models


def per_model_test_acc(models: MLPModels,
                       test_data: torch.Tensor,
                       test_labels: torch.Tensor) -> np.ndarray:
    loss_func = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        _, acc = calculate_loss_acc(test_data, test_labels, models,
                                    loss_func, batch_size=1)
    return acc.cpu().numpy()


def load_stats(base_dir: str,
               sweep: str,
               opt: str,
               init: str,
               sweep_values: list,
               num_samples_list: list,
               test_data: torch.Tensor,
               test_labels: torch.Tensor,
               device: torch.device) -> dict:
    """
    Returns nested dict:
        { num_samples: { sweep_value: (mean_acc, std_acc) } }
    """
    results: dict = {ns: {} for ns in num_samples_list}
    sv_char = "w" if sweep == "width" else "d"

    for sv in sweep_values:
        tag    = f"{opt}_{init}_{sv_char}{sv}"
        folder = os.path.join(base_dir, sweep, tag, "models")
        if not os.path.isdir(folder):
            print(f"  [missing] {folder}")
            continue

        for ns in num_samples_list:
            pattern = os.path.join(folder, f"*_s{ns}_*")
            files   = glob.glob(pattern)
            if not files:
                continue
            try:
                models  = load_checkpoint_models(files[0], device)
                acc_arr = per_model_test_acc(models, test_data, test_labels)
                results[ns][sv] = (float(acc_arr.mean()), float(acc_arr.std()))
                del models
            except Exception as exc:
                print(f"  [error] {files[0]}: {exc}")

    return results


# ── Edge helpers ───────────────────────────────────────────────────────────────

def _linear_edges(values: list) -> np.ndarray:
    """Arithmetic-midpoint bin edges for pcolormesh (linear scale)."""
    v = np.array(values, dtype=float)
    mids = (v[:-1] + v[1:]) / 2.0
    lo   = v[0]  - (v[1]  - v[0])  / 2.0
    hi   = v[-1] + (v[-1] - v[-2]) / 2.0
    return np.concatenate([[lo], mids, [hi]])


def _log_edges(values: list) -> np.ndarray:
    """Geometric-midpoint bin edges for pcolormesh (log scale)."""
    lv   = np.log10(np.array(values, dtype=float))
    mids = (lv[:-1] + lv[1:]) / 2.0
    lo   = lv[0]  - (lv[1]  - lv[0])  / 2.0
    hi   = lv[-1] + (lv[-1] - lv[-2]) / 2.0
    return 10 ** np.concatenate([[lo], mids, [hi]])


# ── Plotting ───────────────────────────────────────────────────────────────────

def load_stats_mock(sweep_values: list,
                   num_samples_list: list,
                   rng: np.random.Generator) -> dict:
    """Return synthetic accuracy values that mimic plausible heatmap structure."""
    results: dict = {ns: {} for ns in num_samples_list}
    base = rng.uniform(0.6, 0.95)
    for si, ns in enumerate(num_samples_list):
        sample_boost = 0.15 * si / max(len(num_samples_list) - 1, 1)
        for wi, sv in enumerate(sweep_values):
            sweep_effect = 0.10 * wi / max(len(sweep_values) - 1, 1)
            noise = rng.normal(0, 0.02)
            acc = float(np.clip(base + sample_boost + sweep_effect + noise, 0.5, 1.0))
            results[ns][sv] = (acc, 0.02)
    return results


def col_deltas_vs_first(Z: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
    """
    For each column i, compute mean(Z[:, i] - Z[:, 0]) across rows,
    with IQR-based outlier removal of the per-row differences.
    Returns array of shape (n_cols,) in accuracy units; col 0 is 0.0.
    NaN where insufficient data.
    """
    n_rows, n_cols = Z.shape
    out = np.full(n_cols, np.nan)
    out[0] = 0.0
    for ci in range(1, n_cols):
        diffs = []
        for ri in range(n_rows):
            if not np.isnan(Z[ri, 0]) and not np.isnan(Z[ri, ci]):
                diffs.append(Z[ri, ci] - Z[ri, 0])
        if len(diffs) < 2:
            continue
        d = np.array(diffs)
        if len(d) >= 4:
            q1, q3 = np.percentile(d, 25), np.percentile(d, 75)
            iqr = q3 - q1
            keep = (d >= q1 - iqr_factor * iqr) & (d <= q3 + iqr_factor * iqr)
            if keep.any():
                d = d[keep]
        out[ci] = float(d.mean())
    return out


def build_error_matrix(stats: dict,
                       sweep_values: list,
                       num_samples_list: list) -> np.ndarray:
    """
    Returns Z of shape (n_samples, n_sweep_values) with test accuracy values.
    Missing cells are NaN.
    """
    Z = np.full((len(num_samples_list), len(sweep_values)), np.nan)
    for si, ns in enumerate(num_samples_list):
        for wi, sv in enumerate(sweep_values):
            entry = stats.get(ns, {}).get(sv)
            if entry is not None:
                Z[si, wi] = entry[0]   # test accuracy
    return Z


def make_figure(base_dir: str,
                sweep: str,
                sweep_values: list,
                x_label: str,
                test_data,
                test_labels,
                device,
                output_folder: str,
                num_samples_list: list,
                plot_name: str,
                log_x: bool = False,
                vmax: float = None,
                mock: bool = False):
    """
    4-row × 2-col figure of heatmaps, plus a shared right-side colorbar.
    """
    n_rows, n_cols = len(INITS), len(OPTIMISERS)

    # Collect all values first so we can set a shared vmax if not given
    all_Z = []

    # Bin edges — same for every subplot
    x_edges = _log_edges(sweep_values) if log_x else _linear_edges(sweep_values)
    # Equal-height rows: use integer indices as y-coordinates
    y_edges = np.arange(len(num_samples_list) + 1) - 0.5

    # ── First pass: compute all matrices ──────────────────────────────────────
    matrices = {}
    rng = np.random.default_rng(42)
    for init in INITS:
        for opt in OPTIMISERS:
            print(f"  [{sweep}] init={init}  opt={opt} …")
            if mock:
                stats = load_stats_mock(sweep_values, num_samples_list, rng)
            else:
                stats = load_stats(base_dir, sweep, opt, init,
                                   sweep_values, num_samples_list,
                                   test_data, test_labels, device)
            Z = build_error_matrix(stats, sweep_values, num_samples_list)
            matrices[(init, opt)] = Z
            all_Z.append(Z[~np.isnan(Z)])

    if vmax is None:
        vmax = float(np.nanmax(np.concatenate(all_Z))) if all_Z else 1.0
    vmin = 0.5
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # ── Figure layout: leave right margin for colorbar ─────────────────────
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(11, 4.5 * n_rows),
                              sharey=True, sharex=True)
    fig.subplots_adjust(left=0.12, right=0.87, wspace=0.08, hspace=0.40)

    for row, init in enumerate(INITS):
        for col, opt in enumerate(OPTIMISERS):
            ax    = axes[row][col]
            Z     = matrices[(init, opt)]

            pcm = ax.pcolormesh(x_edges, y_edges, Z,
                                cmap=CMAP, norm=norm, shading="flat")

            # x-axis scale and ticks
            if log_x:
                ax.set_xscale("log")
                ax.set_xticks(sweep_values)
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            else:
                ax.set_xticks(sweep_values)

            ax.set_yticks(range(len(num_samples_list)))
            ax.set_yticklabels([str(ns) for ns in num_samples_list])
            ax.tick_params(labelsize=22)
            ax.tick_params(axis="x", labelbottom=True, labelsize=22)

            # Per-column Δ annotations above each column (vs. leftmost column)
            col_deltas = col_deltas_vs_first(Z)
            for ci, sv in enumerate(sweep_values):
                d = col_deltas[ci]
                if ci == 0 or np.isnan(d):
                    continue
                sign = "+" if d >= 0 else ""
                ax.annotate(f"{d*100:.1f}%",
                            xy=(sv, 1.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=15, color="black")

            # Column headers (top row only) — pushed above the per-column labels
            if row == 0:
                ax.set_title(OPT_LABELS[opt], fontsize=36, fontweight="bold",
                             pad=60)

            # Row labels (left column only)
            if col == 0:
                ax.set_ylabel(INIT_LABELS[init],
                              fontsize=25)
                ax.yaxis.labelpad = 34
                ax.annotate(
                        "# Samples",
                        xy=(-0.18, 0.75),             # Coordinates: X pushes it left of the axis, Y centers it vertically
                        xycoords="axes fraction",
                        rotation=90,                 # Rotates it vertically to match the y-axis
                        ha="center", 
                        va="top",
                        fontsize=20
                )

            # x-axis label (bottom row only)
            if row == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=25, labelpad=10)

    # ── Shared colorbar ───────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.89, 0.15, 0.025, 0.70])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Test accuracy", fontsize=28, labelpad=10)
    cbar.ax.tick_params(labelsize=20)

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"{plot_name}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output_base",      default=os.path.join(_HERE, "output"))
    p.add_argument("--output_folder",    default=os.path.join(_HERE, "heatmaps"))
    p.add_argument("--widths",           type=int, nargs="+", default=WIDTHS)
    p.add_argument("--depths",           type=int, nargs="+", default=DEPTHS)
    p.add_argument("--num_samples",      type=int, nargs="+",
                   default=NUM_SAMPLES_LIST)
    p.add_argument("--width_plot_name",  default="kink_width_heatmap")
    p.add_argument("--depth_plot_name",  default="kink_depth_heatmap")
    p.add_argument("--vmax",             type=float, default=None,
                   help="Colormap ceiling for test accuracy. "
                        "Default: auto (max observed value per figure).")
    p.add_argument("--test_samples",     type=int, default=100)
    p.add_argument("--test_seed",        type=int, default=0)
    p.add_argument("--device",           default="cpu")
    p.add_argument("--mock", action="store_true",
                   help="Use synthetic random data instead of loading checkpoints "
                        "(instant preview of the figure layout).")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    if args.mock:
        print("[mock mode] skipping test-set construction")
        test_data, test_labels = None, None
    else:
        print("Building fixed test set …")
        test_data, test_labels = get_test_data(args.test_samples,
                                               args.test_seed, device)
        print(f"  test set size: {len(test_data)}")

    # ── Width heatmap ─────────────────────────────────────────────────────────
    print("\nWidth heatmap …")
    make_figure(
        base_dir        = args.output_base,
        sweep           = "width",
        sweep_values    = args.widths,
        x_label         = "Network width",
        test_data       = test_data,
        test_labels     = test_labels,
        device          = device,
        output_folder   = args.output_folder,
        num_samples_list= args.num_samples,
        plot_name       = args.width_plot_name,
        log_x           = True,
        vmax            = args.vmax,
        mock            = args.mock,
    )

    # ── Depth heatmap ─────────────────────────────────────────────────────────
    print("\nDepth heatmap …")
    make_figure(
        base_dir        = args.output_base,
        sweep           = "depth",
        sweep_values    = args.depths,
        x_label         = "Network depth",
        test_data       = test_data,
        test_labels     = test_labels,
        device          = device,
        output_folder   = args.output_folder,
        num_samples_list= args.num_samples,
        plot_name       = args.depth_plot_name,
        log_x           = False,
        vmax            = args.vmax,
        mock            = args.mock,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
