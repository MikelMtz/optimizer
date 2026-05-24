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
import pickle
import zipfile
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
from utils_bias_0 import LeNetModels, calculate_loss_acc

# ── Constants ──────────────────────────────────────────────────────────────────
INITS      = ["uniform", "uniform_02", "kaiming_uniform", "kaiming_normal"]
WIDTHS     = [0.25, 0.5, 1, 2, 4, 8, 16]  # LeNet width multipliers
DEPTHS     = [1, 2, 3, 4]                 # LeNet depth configs
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

def _fmt_sv(sv):
    """Format a sweep value to match bash variable expansion (e.g. 1 not 1.0)."""
    if isinstance(sv, float) and sv == int(sv):
        return str(int(sv))
    return str(sv)


def get_test_data(samples: int, seed: int, device: torch.device):
    """Build a fixed Kink test set embedded as 28×28 single-channel images."""
    kink = Kink(train=False, samples=samples, seed=seed,
                noise=KINK_NOISE, margin=KINK_MARGIN)
    pts         = torch.tensor(kink.data,   dtype=torch.float32).to(device)
    test_labels = torch.tensor(kink.labels, dtype=torch.long).to(device)
    # Embed 2-D kink points into 28×28 single-channel images (same as training)
    N = pts.shape[0]
    test_data = torch.zeros(N, 1, 28, 28, device=device)
    px = ((pts[:, 0] + 1) / 2 * 27).long().clamp(0, 27)
    py = ((pts[:, 1] + 1) / 2 * 27).long().clamp(0, 27)
    test_data[torch.arange(N, device=device), 0, py, px] = 1.0
    return test_data, test_labels


def get_mnist_test_data(seed: int, device: torch.device,
                        num_classes: int = 2, data_root: str = None):
    """Load the MNIST test set for the *num_classes* classes selected by *seed*.

    Replicates the class-selection logic of datasets.MNIST so that the same
    classes are evaluated as were used during training.
    """
    import torchvision
    import torchvision.transforms as tvt
    if data_root is None:
        data_root = os.path.join(_HERE, "..", "data")
    # Standard MNIST statistics — avoids iterating over 60k training images
    mean, std = (0.1307,), (0.3081,)
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize(mean, std)])
    complete_test = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    labels_test = complete_test.targets
    new_labels  = -torch.ones_like(labels_test)
    # Replicate class selection: torch.manual_seed(seed) then randperm(10)
    torch.manual_seed(seed)
    selected = torch.arange(10)[torch.randperm(10)][:num_classes]
    idx_list = []
    for i, cls in enumerate(selected):
        mask = labels_test == cls
        new_labels[mask] = i
        idx_list.append(torch.where(mask)[0])
    complete_test.targets = new_labels
    test_indices = torch.cat(idx_list)
    subset = torch.utils.data.Subset(complete_test, test_indices)
    xs, ys = zip(*[(x[None], y) for x, y in subset])
    return torch.cat(xs).to(device), torch.tensor(ys).to(device)


_LEGACY_DTYPE: dict = {
    'FloatStorage':        torch.float32,
    'DoubleStorage':       torch.float64,
    'HalfStorage':         torch.float16,
    'BFloat16Storage':     torch.bfloat16,
    'LongStorage':         torch.int64,
    'IntStorage':          torch.int32,
    'ShortStorage':        torch.int16,
    'CharStorage':         torch.int8,
    'ByteStorage':         torch.uint8,
    'BoolStorage':         torch.bool,
    'ComplexFloatStorage': torch.complex64,
    'ComplexDoubleStorage':torch.complex128,
}


def _storage_dtype(storage_cls):
    dtype = getattr(storage_cls, 'dtype', None)
    if dtype is not None:
        return dtype
    return _LEGACY_DTYPE.get(storage_cls.__name__, torch.float32)


def _load_via_python_zipfile(path: str) -> dict:
    """
    Fallback for checkpoints that PyTorch's C++ ZIP reader cannot open
    (typically files using ZIP64 extensions, common for ≥ ~2 GB checkpoints).
    Python's zipfile module handles ZIP64 correctly.
    """
    with zipfile.ZipFile(path, 'r') as zf:
        names = zf.namelist()

        # Archive prefix = filename stem, e.g. "kink_s2_lenet_w16.0_.../" or "archive/"
        pkl_name = next(n for n in names if n.endswith('data.pkl'))
        prefix   = pkl_name[:-len('data.pkl')]   # includes trailing '/'
        data_prefix = prefix + 'data/'

        # Read every tensor-storage blob into memory
        raw_storages: dict = {}
        for name in names:
            if name.startswith(data_prefix) and not name.endswith('/'):
                raw_storages[name[len(data_prefix):]] = zf.read(name)

        class _Unpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                assert pid[0] == 'storage', f'Unknown pid type: {pid[0]}'
                _, storage_cls, key, location, numel = pid
                raw = raw_storages[key]
                # Call from_buffer on the storage class itself — avoids all
                # dtype/UntypedStorage compatibility issues across PyTorch versions.
                try:
                    return storage_cls.from_buffer(raw, byte_order=sys.byteorder)
                except TypeError:
                    return storage_cls.from_buffer(raw)

        with zf.open(pkl_name) as pkl:
            return _Unpickler(pkl).load()


def load_checkpoint_models(path: str, device: torch.device) -> LeNetModels:
    # Try the normal path first; fall back to the Python ZIP reader if
    # PyTorch's C++ reader fails (typically a ZIP64 issue on large files).
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        msg = str(e).lower()
        if 'zip' in msg or 'central directory' in msg:
            size_gb = os.path.getsize(path) / 1024 ** 3
            print(f"  [zip fallback {size_gb:.2f} GB] using Python ZIP reader …")
            try:
                ckpt = _load_via_python_zipfile(path)
            except zipfile.BadZipFile:
                raise RuntimeError(
                    "checkpoint is truncated or corrupt "
                    "(ZIP structure invalid — likely a partial write)"
                ) from None
        else:
            raise
    kwargs = dict(ckpt["kwargs"])
    kwargs.pop("device", None)   # LeNetModels has no device arg; use .to() instead
    models = LeNetModels(**kwargs)
    models.load_state_dict(ckpt["good_models_state_dict"])
    return models.to(device)


def per_model_test_acc(models: LeNetModels,
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
               device: torch.device,
               dataset: str = "KINK") -> dict:
    """
    Returns nested dict:
        { num_samples: { sweep_value: (mean_acc, std_acc) } }
    """
    results: dict = {ns: {} for ns in num_samples_list}
    sv_char = "w" if sweep == "width" else "d"

    for sv in sweep_values:
        tag    = f"{opt}_{init}_{sv_char}{_fmt_sv(sv)}"
        if dataset == "MNIST":
            folder = os.path.join(base_dir, tag, "models")
        else:
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
    # Base accuracy rises with samples, varies across sweep values
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


def avg_range_robust(Z: np.ndarray, iqr_factor: float = 1.5) -> float:
    """
    For each row (sample count) in Z compute max-min across sweep values,
    then return the mean after IQR-based outlier removal.
    Returns NaN if fewer than 2 valid rows exist.
    """
    row_ranges = []
    for row in Z:
        valid = row[~np.isnan(row)]
        if len(valid) >= 2:
            row_ranges.append(float(valid.max() - valid.min()))
    if len(row_ranges) < 2:
        return float('nan')
    r = np.array(row_ranges)
    if len(r) >= 4:
        q1, q3 = np.percentile(r, 25), np.percentile(r, 75)
        iqr = q3 - q1
        keep = (r >= q1 - iqr_factor * iqr) & (r <= q3 + iqr_factor * iqr)
        if keep.any():
            r = r[keep]
    return float(r.mean())


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
                dataset: str = "KINK",
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
                                   test_data, test_labels, device,
                                   dataset=dataset)
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
            ax.tick_params(axis="x", labelbottom=True, labelsize=16)

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
                            fontsize=12 if sweep == 'width' else 16, color="black")

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
    p.add_argument("--dataset",           choices=["KINK", "MNIST"], default="KINK",
                   help="Dataset to plot (default: KINK)")
    p.add_argument("--output_base",       default=None,
                   help="Parent of width/ and depth/ folders (KINK) or flat tag "
                        "folders (MNIST). Default: auto-set from --dataset.")
    p.add_argument("--output_folder",     default=os.path.join(_HERE, "heatmaps"))
    p.add_argument("--widths",            type=float, nargs="+", default=WIDTHS)
    p.add_argument("--depths",            type=int,   nargs="+", default=DEPTHS,
                   help="Network depths (KINK only, default: 1 2 3 4)")
    p.add_argument("--num_samples",       type=int, nargs="+",
                   default=NUM_SAMPLES_LIST)
    p.add_argument("--width_plot_name",   default=None,
                   help="Output filename stem for the width heatmap "
                        "(default: {kink,mnist}_lenet_width_heatmap)")
    p.add_argument("--depth_plot_name",   default=None,
                   help="Output filename stem for the depth heatmap "
                        "(default: kink_lenet_depth_heatmap; KINK only)")
    p.add_argument("--vmax",              type=float, default=None,
                   help="Colormap ceiling for test accuracy. "
                        "Default: auto (max observed value per figure).")
    p.add_argument("--test_samples",      type=int, default=100,
                   help="Total Kink samples used to build the test set (KINK only)")
    p.add_argument("--test_seed",         type=int, default=0)
    p.add_argument("--mnist_num_classes", type=int, default=2,
                   help="Number of MNIST classes used during training (MNIST only, default: 2)")
    p.add_argument("--mnist_data_root",   default=None,
                   help="Root directory for MNIST data "
                        "(MNIST only, default: <script_dir>/../../data)")
    p.add_argument("--device",            default="cpu")
    p.add_argument("--mock", action="store_true",
                   help="Use synthetic random data instead of loading checkpoints "
                        "(instant preview of the figure layout).")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # ── Fill in dataset-dependent defaults ────────────────────────────────────
    if args.output_base is None:
        args.output_base = os.path.join(
            _HERE, "mnist_lenet" if args.dataset == "MNIST" else "kink_lenet_sweep"
        )
    prefix = "mnist" if args.dataset == "MNIST" else "kink"
    if args.width_plot_name is None:
        args.width_plot_name = f"{prefix}_lenet_width_heatmap"
    if args.depth_plot_name is None:
        args.depth_plot_name = "kink_lenet_depth_heatmap"

    if args.mock:
        print("[mock mode] skipping test-set construction")
        test_data, test_labels = None, None
    else:
        print("Building fixed test set …")
        if args.dataset == "MNIST":
            test_data, test_labels = get_mnist_test_data(
                args.test_seed, device, args.mnist_num_classes, args.mnist_data_root
            )
        else:
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
        dataset         = args.dataset,
        mock            = args.mock,
    )

    if args.dataset != "MNIST":
        # ── Depth heatmap ─────────────────────────────────────────────────────
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
            dataset         = args.dataset,
            mock            = args.mock,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
