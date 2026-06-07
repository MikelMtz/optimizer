import sqlite3
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, (name, label) in zip(axes, [
    ('mnist_guess', 'GUESS'), ('mnist_sgd', 'SGD'), ('mnist_linear', 'Linear')
]):
    db = f'output/table2/{name}/model_stats.db'
    con = sqlite3.connect(db)
    rows = con.execute("""
        SELECT num_train_samples, loss_bin_l, loss_bin_u, AVG(test_acc)
        FROM model_stats WHERE status='COMPLETE'
        GROUP BY num_train_samples, loss_bin_l, loss_bin_u
    """).fetchall()
    con.close()

    # Group by num_train_samples
    from collections import defaultdict
    data = defaultdict(lambda: ([], []))
    for n, l, u, acc in rows:
        mid = (l + u) / 2
        data[n][0].append(mid)
        data[n][1].append(acc)

    for n in sorted(data.keys()):
        xs, ys = data[n]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], 'o-', label=f'n={n}')

    ax.set_xlabel('Training Loss (bin midpoint)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(label)
    ax.legend()

plt.tight_layout()
plt.savefig('output/table2/table2_plot.png', dpi=150)
print("Saved to output/table2/table2_plot.png")