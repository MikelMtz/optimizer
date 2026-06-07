"""
plot_kink_sample_sweep.py
=========================
Plot test accuracy vs. number of training samples for the kink experiment.

Produces two sets of figures:
  - Width sweep:  4 plots (one per init), each with 5 widths × 2 optimisers
  - Depth sweep:  4 plots (one per init), each with 6 depths × 2 optimisers

Each line = one (optimiser, sweep-value) pair; shaded band = ±1 std across
the 100 perfect models stored in the checkpoint.

Usage (run from check_bias_0/):
    python plot_kink_sample_sweep.py

Options:
    --output_base   Parent of width/ and depth/ folders
                    [default: output/kink_sample_sweep]
    --output_folder Where to save plots [default: plots/kink_sample_sweep]
    --test_samples  Total samples used to build the fixed test set (30% are
                    test points) [default: 100 → 30 test points]
    --test_seed     Seed for the fixed test set [default: 0] 
    --device        torch device [default: cpu] 
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

# Allow imports from parent directory (MLPModels) and from current dir (datasets)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from datasets import Kink
from utils_bias_0 import MLPModels, calculate_loss_acc

# ── Constants ──────────────────────────────────────────────────────────────────
INITS = ["uniform", "uniform_02", "kaiming_uniform", "kaiming_normal"]
WIDTHS = [2, 8, 32, 128, 512]
DEPTHS = [1, 2, 3, 4, 5, 6]
NUM_SAMPLES_LIST = [2, 4, 6, 8, 12, 16, 20, 26, 30]

INIT_LABELS = {
    "uniform":         "$Uniform [-1,1]$",
    "uniform_02":      "$Uniform [-0.2, 0.2]$",
    "kaiming_uniform": "Kaiming Uniform",
    "kaiming_normal":  "Kaiming Normal",
}

KINK_MARGIN = 0.25
KINK_NOISE  = 0.0

# Colour maps: guess → blue family; sgd → red/orange family
# We map sweep-value index to a scalar in [0.35, 0.90] within the map
_GUESS_CMAP = plt.cm.Blues
_SGD_CMAP   = plt.cm.Reds

OPT_DISPLAY = {"guess": "G&C", "sgd": "SGD"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_test_data(samples: int, seed: int, device: torch.device):
    """Build a fixed Kink test set (30 % of *samples* points)."""
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
    """Return array of shape (model_count,) with per-model test accuracy."""
    loss_func = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        _, acc = calculate_loss_acc(test_data, test_labels, models,
                                    loss_func, batch_size=1)
    return acc.cpu().numpy()


def sweep_stats(base_dir: str,
                sweep: str,
                opt: str,
                init: str,
                sweep_values: list,
                num_samples_list: list,
                test_data: torch.Tensor,
                test_labels: torch.Tensor,
                device: torch.device):
    """
    Returns a dict:
      { sweep_value: { num_samples: (mean_acc, std_acc) } }
    """
    results = {}
    for sv in sweep_values:
        tag    = f"{opt}_{init}_{'w' if sweep=='width' else 'd'}{sv}"
        folder = os.path.join(base_dir, sweep, tag, "models")
        if not os.path.isdir(folder):
            print(f"  [missing] {folder}")
            continue

        sv_dict = {}
        for ns in num_samples_list:
            # checkpoint names contain _s{ns}_ in the filename
            pattern = os.path.join(folder, f"*_s{ns}_*")
            files   = glob.glob(pattern)
            if not files:
                continue
            try:
                models  = load_checkpoint_models(files[0], device)
                acc_arr = per_model_test_acc(models, test_data, test_labels)
                sv_dict[ns] = (float(acc_arr.mean()), float(acc_arr.std()))
                del models
            except Exception as exc:
                print(f"  [error] {files[0]}: {exc}")
        results[sv] = sv_dict
    return results


def _palette(cmap, n: int):
    """Return *n* colours evenly spaced in [0.35, 0.90] of *cmap*."""
    lo, hi = 0.35, 0.90
    if n == 1:
        return [cmap(0.65)]
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def draw_lines(ax, stats: dict, sweep_values: list, num_samples_list: list,
               cmap, sv_labels: list, opt_label: str):
    """Draw one set of lines (one optimizer) on *ax*."""
    n      = len(sweep_values)
    colors = _palette(cmap, n)

    for idx, sv in enumerate(sweep_values):
        if sv not in stats:
            continue
        sv_dict = stats[sv]
        xs      = sorted(sv_dict.keys())
        means   = np.array([sv_dict[ns][0] for ns in xs])
        stds    = np.array([sv_dict[ns][1] for ns in xs])

        lbl = f"{opt_label} {sv_labels[idx]}"
        ax.plot(xs, means, color=colors[idx], lw=1.8, zorder=2)
        ax.errorbar(xs, means, yerr=stds,
                    fmt="o", color=colors[idx], markersize=6.0,
                    elinewidth=1.0, capsize=3.5, capthick=1.0,
                    label=lbl, zorder=3)


def make_figure(base_dir, sweep, sweep_values, sv_label_fn,
                test_data, test_labels, device, output_folder,
                plot_name=None):
    """Create and save 4-subplot figure (2×2) for one sweep type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig.subplots_adjust(wspace=0.10, hspace=0.38)

    for idx, init in enumerate(INITS):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        print(f"  [{sweep}] init={init} …")

        stats_guess = sweep_stats(base_dir, sweep, "guess", init,
                                   sweep_values, NUM_SAMPLES_LIST,
                                   test_data, test_labels, device)
        stats_sgd   = sweep_stats(base_dir, sweep, "sgd",   init,
                                   sweep_values, NUM_SAMPLES_LIST,
                                   test_data, test_labels, device)

        sv_labels = [sv_label_fn(sv) for sv in sweep_values]

        draw_lines(ax, stats_guess, sweep_values, NUM_SAMPLES_LIST,
                   _GUESS_CMAP, sv_labels, "G&C")
        draw_lines(ax, stats_sgd,   sweep_values, NUM_SAMPLES_LIST,
                   _SGD_CMAP,   sv_labels, "SGD")

        # Reference line at perfect accuracy
        ax.axhline(1.0, color="silver", linestyle="--", lw=0.8, zorder=0)

        ax.set_title(INIT_LABELS[init], fontsize=13, fontweight="bold")
        ax.set_xlabel("# samples", fontsize=11)
        if col == 0:
            ax.set_ylabel("Test accuracy", fontsize=11)
        ax.set_ylim(0.5, 1.05)
        ax.set_xticks([2, 6, 10, 14, 18, 22, 26, 30])
        ax.tick_params(labelsize=10)

        # Legend on every bottom-right subplot (last in each row)
        if col == 1:
            handles, labels_leg = ax.get_legend_handles_labels()
            ax.legend(handles, labels_leg, fontsize=9,
                      loc="lower right", ncol=2)

    os.makedirs(output_folder, exist_ok=True)
    stem      = plot_name if plot_name else f"kink_{sweep}_sweep"
    save_path = os.path.join(output_folder, f"{stem}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_base",      default=os.path.join(_HERE, "output"))
    p.add_argument("--output_folder",    default=os.path.join(_HERE, "plots"))
    p.add_argument("--widths",           type=int, nargs="+", default=WIDTHS,
                   help="Hidden-unit widths to plot (default: 2 8 32 128 512)")
    p.add_argument("--depths",           type=int, nargs="+", default=DEPTHS,
                   help="Network depths to plot (default: 1 2 3 4 5 6)")
    p.add_argument("--width_plot_name",  default="kink_width_sweep",
                   help="Output filename stem for the width sweep figure")
    p.add_argument("--depth_plot_name",  default="kink_depth_sweep",
                   help="Output filename stem for the depth sweep figure")
    p.add_argument("--test_samples",     type=int, default=100)
    p.add_argument("--test_seed",        type=int, default=0)
    p.add_argument("--device",           default="cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print("Building fixed test set …")
    test_data, test_labels = get_test_data(args.test_samples,
                                           args.test_seed, device)
    print(f"  test set size: {len(test_data)}")

    # ── Width sweep ───────────────────────────────────────────────────────────
    print("\nWidth sweep …")
    make_figure(
        base_dir      = args.output_base,
        sweep         = "width",
        sweep_values  = args.widths,
        sv_label_fn   = lambda w: f"w={w}",
        test_data     = test_data,
        test_labels   = test_labels,
        device        = device,
        output_folder = args.output_folder,
        plot_name     = args.width_plot_name,
    )

    # ── Depth sweep ───────────────────────────────────────────────────────────
    print("\nDepth sweep …")
    make_figure(
        base_dir      = args.output_base,
        sweep         = "depth",
        sweep_values  = args.depths,
        sv_label_fn   = lambda d: f"d={d}",
        test_data     = test_data,
        test_labels   = test_labels,
        device        = device,
        output_folder = args.output_folder,
        plot_name     = args.depth_plot_name,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
