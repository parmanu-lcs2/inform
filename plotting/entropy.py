import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "emergence_data"
TASKS = ["mmlu", "gsm8k", "humaneval"]
OUT_DIR = "emergence_plots"
os.makedirs(OUT_DIR, exist_ok=True)

BOOTSTRAP_SAMPLES = 1000
CI_ALPHA = 0.05  # 95% CI

# Metrics
def entropy(p, eps=1e-9):
    p = p / p.sum(dim=-1, keepdim=True)
    return -(p * torch.log(p + eps)).sum(dim=-1)

def gini(x):
    x = x.flatten().numpy()
    x = np.sort(x)
    n = len(x)
    return (2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n

def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        means.append(sample.mean())
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi

for task in TASKS:
    print(f"\nProcessing task: {task}")
    task_dir = os.path.join(BASE_DIR, task)
    if not os.path.isdir(task_dir):
        print(f"[warning] Missing directory: {task_dir}")
        continue

    rows = []
    prev_C = None
    all_C_epochs = []

    files = sorted(f for f in os.listdir(task_dir) if f.endswith(".pt"))

    for f in files:
        epoch = int(f.split("epoch")[-1].split(".")[0])
        data = torch.load(os.path.join(task_dir, f), map_location="cpu")

        C = data["C"]
        seq = data["seq"]

        all_C_epochs.append(C)

        collab_entropy_samples = entropy(C).mean(dim=1).numpy()
        seq_entropy_samples = entropy(seq).numpy()

        incoming = C.sum(dim=1) 
        centralization_samples = np.array([gini(x) for x in incoming])

        collab_entropy = collab_entropy_samples.mean()
        seq_entropy = seq_entropy_samples.mean()
        centralization = centralization_samples.mean()
        rank = torch.linalg.matrix_rank(C.mean(dim=0)).item()

        ce_lo, ce_hi = bootstrap_ci(collab_entropy_samples)
        se_lo, se_hi = bootstrap_ci(seq_entropy_samples)
        cen_lo, cen_hi = bootstrap_ci(centralization_samples)

        if prev_C is not None:
            kl = torch.sum(
                C * (torch.log(C + 1e-9) - torch.log(prev_C + 1e-9)),
                dim=(1, 2)
            ).mean().item()
        else:
            kl = 0.0

        prev_C = C.clone()

        rows.append({
            "task": task,
            "epoch": epoch,
            "collab_entropy": collab_entropy,
            "collab_entropy_lo": ce_lo,
            "collab_entropy_hi": ce_hi,
            "seq_entropy": seq_entropy,
            "seq_entropy_lo": se_lo,
            "seq_entropy_hi": se_hi,
            "centralization": centralization,
            "centralization_lo": cen_lo,
            "centralization_hi": cen_hi,
            "rank": rank,
            "kl_drift": kl
        })

        avg_C = C.mean(dim=0)
        plt.figure(figsize=(6, 5))
        sns.heatmap(avg_C.numpy(), cmap="viridis", square=True)
        plt.title(f"{task.upper()} – Collaboration Matrix (Epoch {epoch})")
        plt.xlabel("Target Expert")
        plt.ylabel("Source Expert")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, f"{task}_epoch{epoch}_collab_heatmap.pdf")
        )
        plt.close()

    df = pd.DataFrame(rows).sort_values("epoch")
    csv_path = os.path.join(OUT_DIR, f"{task}_emergence_metrics_ci.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    TASK_COLORS = {
        "gsm8k": "#1f77b4",
        "mmlu": "#d62728", 
        "humaneval": "#2ca02c",
    }

    def plot_with_ci(x, y, lo, hi, ylabel, task, title, fname):
        x = [str(i) for i in x.tolist()]
        color = TASK_COLORS.get(task.lower(), "black")
        task_label = task.upper()

        plt.figure(figsize=(8, 6))

        plt.plot(
            x, y,
            marker="o",
            color=color,
            linewidth=2,
            label=task_label
        )

        plt.fill_between(
            x, lo, hi,
            color=color,
            alpha=0.3
        )

        plt.xlabel("Epoch", fontsize=30)
        plt.ylabel(ylabel, fontsize=30)

        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.legend(fontsize=40, frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname))
        plt.close()

    plot_with_ci(
        df.epoch,
        df.collab_entropy,
        df.collab_entropy_lo,
        df.collab_entropy_hi,
        "Collaboration Entropy",
        task,
        f"{task.upper()} - Emergence of Routing Confidence",
        f"{task}_collab_entropy_ci.pdf"
    )

    plot_with_ci(
        df.epoch,
        df.seq_entropy,
        df.seq_entropy_lo,
        df.seq_entropy_hi,
        "Sequence Entropy",
        task,
        f"{task.upper()} - Emergence of Ordering",
        f"{task}_seq_entropy_ci.pdf"
    )

    plot_with_ci(
        df.epoch,
        df.centralization,
        df.centralization_lo,
        df.centralization_hi,
        "Gini (Incoming Mass)",
        task,
        f"{task.upper()} - Refiner Centralization",
        f"{task}_centralization_ci.pdf"
    )

    full_C = torch.cat(all_C_epochs, dim=0)
    avg_C_all = full_C.mean(dim=0)

    labels = [f"{i+1}" for i in range(avg_C_all.size(0))]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
                avg_C_all.numpy(), 
                annot=True, 
                fmt=".2f", 
                cmap="YlGnBu", 
                xticklabels=labels, 
                yticklabels=labels,
                annot_kws={"fontsize":12},
                vmin=0, 
                vmax=1, 
                cbar=False
            )
    plt.xlabel("Target Expert")
    plt.ylabel("Source Expert")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"{task}_collab_heatmap_avg.pdf")
    )
    plt.close()

print("\n[DONE] CI plots and heatmaps generated for all tasks.")
