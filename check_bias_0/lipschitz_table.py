"""Generate a summary table for the Lipschitz experiment.

For each (hidden_units, init, optimizer) combination, loads the saved models,
evaluates per-model test accuracy on the Kink dataset, and reports:
  - mean test accuracy  (over successfully trained networks)
  - variance of test accuracy

Usage:
    python scripts/lipschitz_table.py [--output_base output/lipschitz]
"""
import os
import sys
import glob
import argparse
import torch
import torch.nn as nn

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import MLPModels, calculate_loss_acc
from datasets import Kink

WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
INITS = ["uniform", "uniform_02", "kaiming_uniform", "kaiming_normal"]
OPTIMIZERS = ["guess", "SGD"]

INIT_DISPLAY = {
    "uniform":          "Uniform[-1,1]",
    "uniform_02":       "Uniform[-0.2,0.2]",
    "kaiming_uniform":  "Kaiming Uniform",
    "kaiming_normal":   "Kaiming Normal",
}

parser = argparse.ArgumentParser()
parser.add_argument("--output_base", default="output/lipschitz",
                    help="Base folder that contains the per-run output directories")
parser.add_argument("--test_samples", type=int, default=100,
                    help="Number of test samples for evaluation")
parser.add_argument("--test_seed", type=int, default=0,
                    help="Seed for the test dataset")
args = parser.parse_args()

# Build a large Kink test set for evaluation
test_ds = Kink(train=False, samples=args.test_samples, seed=args.test_seed,
               noise=0.0, margin=0.25)
test_data   = torch.tensor(test_ds.data).float()
test_labels = torch.tensor(test_ds.labels).long()
loss_func   = nn.CrossEntropyLoss(reduction="none")

rows = []  # (width, init, optimizer, mean_acc, var_acc, model_count)

for opt in OPTIMIZERS:
    for init in INITS:
        for w in WIDTHS:
            tag = f"{opt.lower()}_{init}_u{w}"
            model_dir = os.path.join(args.output_base, tag, "models")
            if not os.path.isdir(model_dir):
                rows.append((w, init, opt, None, None, 0))
                continue

            model_files = sorted(glob.glob(os.path.join(model_dir, "*")))
            if len(model_files) == 0:
                rows.append((w, init, opt, None, None, 0))
                continue

            all_test_accs = []
            for mf in model_files:
                try:
                    ckpt = torch.load(mf, map_location="cpu")
                except Exception:
                    continue
                kwargs = ckpt["kwargs"]
                models = MLPModels(**kwargs, device=torch.device("cpu"))
                models.load_state_dict(ckpt["good_models_state_dict"])

                with torch.no_grad():
                    _, test_accs = calculate_loss_acc(
                        test_data, test_labels, models, loss_func, batch_size=1
                    )
                all_test_accs.append(test_accs)

            if len(all_test_accs) == 0:
                rows.append((w, init, opt, None, None, 0))
                continue

            all_test_accs = torch.cat(all_test_accs)
            mean_acc = all_test_accs.mean().item()
            var_acc  = all_test_accs.var().item()
            count    = all_test_accs.numel()
            rows.append((w, init, opt, mean_acc, var_acc, count))

# --- Pretty-print the table ---
header = f"{'Width':>6}  {'Initialization':<22}  {'Optimizer':<8}  {'Mean Test Acc':>14}  {'Var Test Acc':>13}  {'#Models':>8}"
sep = "-" * len(header)
print(sep)
print(header)
print(sep)
for w, init, opt, mean_acc, var_acc, count in rows:
    init_str = INIT_DISPLAY.get(init, init)
    if mean_acc is not None:
        print(f"{w:>6}  {init_str:<22}  {opt:<8}  {mean_acc:>14.6f}  {var_acc:>13.6f}  {count:>8}")
    else:
        print(f"{w:>6}  {init_str:<22}  {opt:<8}  {'N/A':>14}  {'N/A':>13}  {count:>8}")
print(sep)
