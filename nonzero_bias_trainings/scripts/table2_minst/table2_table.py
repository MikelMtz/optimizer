"""
Generate a table (as image) of test accuracy vs training loss bins,
matching the format from the paper:
  Rows: grouped by Sample Count, with sub-rows for each optimizer
  Columns: Sample Count | Arch | Optimizer | Best Test Acc | (0.3,0.35) | ... | (0.6,0.65)
  Cells: mean ± std (%), best per column in bold
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--db_dir', default='output/table2', help='Directory containing mnist_guess/, mnist_sgd/, mnist_linear/ subfolders')
parser.add_argument('--save_path', default='output/table2/table2_table.png', help='Output path for the table image')
args = parser.parse_args()

# ── Configuration ──
loss_bins = [
    (0.3, 0.35), (0.35, 0.4), (0.4, 0.45), (0.45, 0.5),
    (0.5, 0.55), (0.55, 0.6), (0.6, 0.65)
]

optimizers_config = [
    ('mnist_guess',  'LeNet', 'G&C'),
    ('mnist_sgd',    'LeNet', 'SGD'),
    ('mnist_linear', 'Linear', 'Linear'),
]

sample_counts = [32, 16, 8, 4, 2]

# ── Query data ──
# results[(db_name, n_samples, bin_l, bin_u)] = list of test_acc values
results = {}

for db_name, arch, opt_label in optimizers_config:
    db_path = f'{args.db_dir}/{db_name}/model_stats.db'
    con = sqlite3.connect(db_path)
    rows = con.execute("""
        SELECT num_train_samples, loss_bin_l, loss_bin_u, test_acc
        FROM model_stats WHERE status='COMPLETE'
    """).fetchall()
    con.close()
    for n, l, u, acc in rows:
        key = (db_name, n, l, u)
        if key not in results:
            results[key] = []
        results[key].append(acc)

# ── Build table data ──
def fmt_acc(vals):
    """Format as 'mean±std%'"""
    if not vals:
        return '-'
    mean = np.mean(vals) * 100
    std = np.std(vals) * 100
    return f'{mean:.2f}±{std:.2f}%'

def get_mean(vals):
    if not vals:
        return None
    return np.mean(vals) * 100

# Column headers
bin_headers = [f'({l}, {u})' for l, u in loss_bins]
col_headers = ['Sample\nCount', 'Arch', 'Optimizer', 'Best Test Acc'] + bin_headers

# Build rows
table_rows = []  # list of (sample_count, arch, opt_label, best_acc_str, [bin_strs])
table_means = []  # parallel list of numeric means for bolding

for n in sample_counts:
    for db_name, arch, opt_label in optimizers_config:
        bin_strs = []
        bin_means = []
        for (l, u) in loss_bins:
            key = (db_name, n, l, u)
            vals = results.get(key, [])
            bin_strs.append(fmt_acc(vals))
            bin_means.append(get_mean(vals))

        # Best test acc = max mean across all bins
        valid_means = [m for m in bin_means if m is not None]
        if valid_means:
            best = max(valid_means)
            best_idx = [i for i, m in enumerate(bin_means) if m is not None and abs(m - best) < 1e-9][0]
            best_vals = results.get((db_name, n, loss_bins[best_idx][0], loss_bins[best_idx][1]), [])
            best_str = fmt_acc(best_vals)
            best_mean = best
        else:
            best_str = '-'
            best_mean = None

        row = [str(n), arch, opt_label, best_str] + bin_strs
        means = [best_mean] + bin_means
        table_rows.append(row)
        table_means.append(means)

# ── Find best (highest mean) per column within each sample count group ──
# Each group has 3 rows (one per optimizer)
n_opts = len(optimizers_config)
bold_mask = []  # same shape as table_means, True if this cell should be bold

for group_start in range(0, len(table_rows), n_opts):
    group_means = table_means[group_start:group_start + n_opts]
    n_cols = len(group_means[0])
    for row_i in range(n_opts):
        row_bold = []
        for col_j in range(n_cols):
            val = group_means[row_i][col_j]
            if val is None:
                row_bold.append(False)
                continue
            # Check if this is the max in the column for this group
            col_vals = [group_means[r][col_j] for r in range(n_opts) if group_means[r][col_j] is not None]
            if col_vals and abs(val - max(col_vals)) < 1e-9:
                row_bold.append(True)
            else:
                row_bold.append(False)
        bold_mask.append(row_bold)

# ── Render as LaTeX table ──

def fmt_acc_latex(vals):
    """Format as '$mean\\pm std\\%$', return '-' if empty."""
    if not vals:
        return '-'
    mean = np.mean(vals) * 100
    std = np.std(vals) * 100
    return f'${mean:.2f}\\pm{std:.2f}\\%$'

def fmt_acc_latex_bold(vals):
    """Format as bold '$\\mathbf{mean\\pm std\\%}$'."""
    if not vals:
        return '-'
    mean = np.mean(vals) * 100
    std = np.std(vals) * 100
    return f'$\\mathbf{{{mean:.2f}\\pm{std:.2f}\\%}}$'

n_rows = len(table_rows)

# Build LaTeX output
bin_col_spec = 'c' * len(loss_bins)
header_line = (
    '\\begin{table}[ht]\n'
    '\\centering\n'
    '\\caption{Test Accuracy by Training Loss Bin}\n'
    '\\label{tab:table2}\n'
    '\\resizebox{\\textwidth}{!}{\n'
    f'\\begin{{tabular}}{{cc c c |{bin_col_spec}}}\n'
    '\\toprule\n'
)

# Column headers
bin_hdrs = ' & '.join([f'({l}, {u})' for l, u in loss_bins])
header_line += (
    '\\textbf{Sample} & \\textbf{Arch} & \\textbf{Optimizer} & '
    '\\textbf{Best Test Acc} & '
    f'\\multicolumn{{{len(loss_bins)}}}{{c}}{{\\textbf{{Train Loss}}}} \\\\\n'
)
header_line += (
    '\\textbf{Count} & & & & '
    f'{bin_hdrs} \\\\\n'
    '\\midrule\n'
)

print(header_line, end='')

for i, (row, means, bolds) in enumerate(zip(table_rows, table_means, bold_mask)):
    n_str, arch, opt_label, best_str, *bin_strs = row

    # Sample count: only print on first row of group, use multirow
    if i % n_opts == 0:
        sample_cell = f'\\multirow{{{n_opts}}}{{*}}{{{n_str}}}'
    else:
        sample_cell = ''

    # Best test acc cell
    best_vals_key = None
    for bi, (l, u) in enumerate(loss_bins):
        db_name = optimizers_config[[oc[2] for oc in optimizers_config].index(opt_label)][0]
        key = (db_name, int(n_str if n_str else row[0]), l, u)
        if key in results and means[0] is not None:
            m = get_mean(results[key])
            if m is not None and abs(m - means[0]) < 1e-9:
                best_vals_key = key
                break

    if bolds[0] and best_vals_key:
        best_cell = fmt_acc_latex_bold(results[best_vals_key])
    elif best_vals_key:
        best_cell = fmt_acc_latex(results[best_vals_key])
    else:
        best_cell = '-'

    # Loss bin cells
    bin_cells = []
    for bi, (l, u) in enumerate(loss_bins):
        db_name = optimizers_config[[oc[2] for oc in optimizers_config].index(opt_label)][0]
        key = (db_name, int(n_str if n_str else row[0]), l, u)
        vals = results.get(key, [])
        if bolds[bi + 1]:  # +1 because bolds[0] is best_acc
            bin_cells.append(fmt_acc_latex_bold(vals))
        else:
            bin_cells.append(fmt_acc_latex(vals))

    cells = [sample_cell, arch, opt_label, best_cell] + bin_cells
    line = ' & '.join(cells) + ' \\\\\n'
    print(line, end='')

    # Add hline between sample count groups
    if (i + 1) % n_opts == 0 and i + 1 < n_rows:
        print('\\midrule')

# Footer
print(
    '\\bottomrule\n'
    '\\end{tabular}\n'
    '}\n'
    '\\end{table}'
)
