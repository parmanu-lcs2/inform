import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sns.set_context(
    "paper",
    rc={
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16
    }
)


ATTR_DIR = "attribution_data"
OUT_DIR = "attribution_plots"

os.makedirs(OUT_DIR, exist_ok=True)

TASKS = ["mmlu", "gsm8k", "humaneval", ]

sns.set(style="whitegrid", font_scale=1.1)

# SAMPLE-LEVEL HEATMAPS
for task in TASKS:
    task_dir = os.path.join(ATTR_DIR, task)
    files = sorted(glob.glob(os.path.join(task_dir, f"{task}_epoch*.pt")))

    for f in files:
        print(f)
        epoch = int(f.split("epoch")[-1].split(".")[0])
        data = torch.load(f)

        grad = data["grad_attr"].numpy()    # [S, N]
        route = data["route_attr"].numpy()  # [S, N]

        # Gradient Attribution Heatmap
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            grad,
            cmap="viridis",
            cbar_kws={"label": "Gradient Attribution"},
            annot=True, fmt=".2f",
            annot_kws={"fontsize":25},
            cbar=False
        )
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=25)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=25)
        ax.set_xlabel("Expert", fontsize=30)
        ax.set_ylabel("Sample", fontsize=30)
        plt.title(f"EPOCH {epoch}", fontsize=30)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUT_DIR, f"{task}_grad_attr_heatmap_epoch{epoch}.pdf"
            )
        )
        plt.close()

        # Routing Attribution Heatmap
        plt.figure(figsize=(10, 6))
        ax  = sns.heatmap(
            route,
            cmap="magma",
            cbar_kws={"label": "Incoming Routing Mass"},
            annot=True, fmt=".2f",
            annot_kws={"fontsize":25},
            cbar=False
        )
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=25)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=25)
        ax.set_xlabel("Expert", fontsize=30)
        ax.set_ylabel("Sample", fontsize=30)
        plt.title(f"EPOCH {epoch}", fontsize=30)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUT_DIR, f"{task}_route_attr_heatmap_epoch{epoch}.pdf"
            )
        )
        plt.close()

# EPOCH-LEVEL HEATMAPS
for task in TASKS:
    task_dir = os.path.join(ATTR_DIR, task)
    files = sorted(glob.glob(os.path.join(task_dir, f"{task}_epoch*.pt")))

    epochs = []
    grad_means = []
    route_means = []

    for f in files:
        epoch = int(f.split("epoch")[-1].split(".")[0])
        data = torch.load(f)

        grad = data["grad_attr"].numpy()
        route = data["route_attr"].numpy()

        epochs.append(epoch)
        grad_means.append(grad.mean(axis=0))
        route_means.append(route.mean(axis=0))

    grad_means = np.stack(grad_means, axis=0)
    route_means = np.stack(route_means, axis=0)

    # Gradient Attribution Emergence
    labels = [f"{i+1}" for i in range(grad_means.shape[1])]
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        grad_means,
        yticklabels=epochs,
        xticklabels=labels,
        annot=True, fmt=".2f",
        cmap="YlGnBu", 
        annot_kws={"fontsize":20},
        cbar=False
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    ax.set_xlabel("Expert", fontsize=20)
    ax.set_ylabel("Epoch", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"{task}_grad_attr_emergence.pdf")
    )
    plt.close()

    # Routing Attribution Emergence
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        route_means,
        yticklabels=epochs,
        xticklabels=labels,
        annot=True, fmt=".2f",
        cmap="YlOrBr", 
        annot_kws={"fontsize":20},
        cbar=False
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    ax.set_xlabel("Expert", fontsize=20)
    ax.set_ylabel("Epoch", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"{task}_route_attr_emergence.pdf")
    )
    plt.close()

# CORRELATION OVER EPOCHS
for task in TASKS:
    task_dir = os.path.join(ATTR_DIR, task)
    files = sorted(glob.glob(os.path.join(task_dir, f"{task}_epoch*.pt")))

    epochs, rhos = [], []

    for f in files:
        epoch = int(f.split("epoch")[-1].split(".")[0])
        data = torch.load(f)

        grad = data["grad_attr"].mean(axis=0)
        route = data["route_attr"].mean(axis=0)

        rho, _ = spearmanr(grad, route)
        epochs.append(epoch)
        rhos.append(rho)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, rhos, marker="o")
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman ρ")
    plt.title(f"{task.upper()} – Alignment of Intrinsic and Relational Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"{task}_attr_route_corr.pdf")
    )
    plt.close()

print("[DONE] Attribution heatmaps saved to", OUT_DIR)
