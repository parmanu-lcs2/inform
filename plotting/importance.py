import os
import glob
import torch
import argparse
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def load_epoch_file(pt_file):
    data = torch.load(pt_file, map_location="cpu")
    grad_attr = data["grad_attr"] 
    route_attr = data["route_attr"]
    return grad_attr, route_attr

# Intrinsic Expert Attribution
def plot_intrinsic(grad_attr, out_path, title):
    mean_attr = grad_attr.mean(dim=0)
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(mean_attr)), mean_attr.numpy())
    plt.xlabel("Expert Index")
    plt.ylabel("Mean Gradient Attribution")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Relational Expert Importance
def plot_relational(route_attr, out_path, title):
    mean_route = route_attr.mean(dim=0)
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(mean_route)), mean_route.numpy())
    plt.xlabel("Expert Index")
    plt.ylabel("Mean Incoming Routing Mass")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Intrinsic vs Relational
def plot_intrinsic_vs_relational(grad_attr, route_attr, out_path, title):
    grad = grad_attr.mean(dim=0)
    route = route_attr.mean(dim=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(len(grad)), grad.numpy())
    axes[0].set_title("Intrinsic Attribution")
    axes[0].set_xlabel("Expert Index")
    axes[0].set_ylabel("Gradient Norm")
    axes[1].bar(range(len(route)), route.numpy())
    axes[1].set_title("Relational Importance")
    axes[1].set_xlabel("Expert Index")
    axes[1].set_ylabel("Incoming Routing Mass")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Attribution–Routing Alignment over Epochs
def plot_alignment(pt_files, out_path, task):
    epochs = []
    correlations = []

    for pt in pt_files:
        epoch = int(pt.split("epoch")[-1].split(".")[0])
        grad_attr, route_attr = load_epoch_file(pt)

        grad_mean = grad_attr.mean(dim=0)
        route_mean = route_attr.mean(dim=0)
        rho, _ = spearmanr(
            grad_mean.numpy(),
            route_mean.numpy()
        )
        epochs.append(epoch)
        correlations.append(rho)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, correlations, marker="o")
    plt.axhline(0, linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Spearman ρ")
    plt.title(f"{task.upper()}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(args):
    tasks = ["gsm8k", "humaneval", "mmlu"]
    os.makedirs(args.out_dir, exist_ok=True)
    for task in tasks:
        task_files = sorted(
            glob.glob(os.path.join(args.attr_dir, task, f"{task}_epoch*.pt"))
        )

        if not task_files:
            print(f"[skip] No files found for {task}")
            continue

        print(f"[info] {task}: {len(task_files)} epochs")

        final_pt = task_files[-1]
        grad_attr, route_attr = load_epoch_file(final_pt)

        plot_intrinsic(
            grad_attr,
            os.path.join(args.out_dir, f"{task}_intrinsic_attr.pdf"),
            f"{task.upper()} – Intrinsic Expert Attribution"
        )

        plot_relational(
            route_attr,
            os.path.join(args.out_dir, f"{task}_relational_attr.pdf"),
            f"{task.upper()} – Relational Expert Importance"
        )

        plot_intrinsic_vs_relational(
            grad_attr,
            route_attr,
            os.path.join(args.out_dir, f"{task}_intrinsic_vs_relational.pdf"),
            f"{task.upper()} – Intrinsic vs Relational Importance"
        )

        plot_alignment(
            task_files,
            os.path.join(args.out_dir, f"{task}_attr_route_alignment.pdf"),
            task
        )

        print(f"[saved] Plots for {task}")

    print("\n[DONE] Expert attribution plots generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attr_dir",
        required=True,
        help="Directory containing attribution .pt files"
    )

    parser.add_argument(
        "--out_dir",
        default="attribution_plots",
        help="Directory to save plots"
    )

    args = parser.parse_args()
    main(args)
