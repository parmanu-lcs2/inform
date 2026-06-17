import glob
import os
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set_context(
    "paper",
    rc={
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
)

def plot_collaboration_matrices(glob_pattern, output_dir="plots"):
    """
    Plots collaboration matrices found by the glob pattern.
    Saves ONE image file per Task/Epoch pair.
    """
    # 1. Find files
    files = glob.glob(glob_pattern)
    files.sort()
    
    if not files:
        print(f"No files found matching pattern: {glob_pattern}")
        return

    print(f"Found {len(files)} matrix files. Processing...")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Parse Filenames and Organize Data
    filename_re = re.compile(r"avg_C_matrix_(.+?)_epoch(\d+)_")
    
    data = {} 

    for f_path in files:
        basename = os.path.basename(f_path)
        match = filename_re.search(basename)
        
        if match:
            task = match.group(1)
            epoch = int(match.group(2))
            
            try:
                matrix = torch.load(f_path, map_location='cpu')
                if isinstance(matrix, torch.Tensor):
                    matrix = matrix.float().cpu().numpy()
                
                if task not in data:
                    data[task] = {}
                data[task][epoch] = matrix
            except Exception as e:
                print(f"Error loading {basename}: {e}")

    # 3. Generate Individual Plots
    for task, epochs_data in data.items():
        sorted_epochs = sorted(epochs_data.keys())
        
        # Determine Axis Labels from the first matrix available
        first_matrix = epochs_data[sorted_epochs[0]]
        num_models = first_matrix.shape[0]
        
        labels = [f"{i+1}" for i in range(num_models)]

        for epoch in sorted_epochs:
            matrix = epochs_data[epoch]
            
            # Create a single figure for this epoch
            plt.figure(figsize=(8, 6))
            
            ax = sns.heatmap(
                matrix,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={"fontsize": 16},
                vmin=0,
                vmax=1,
                cbar=False
            )

            ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
            
            plt.title(f"Collaboration Matrix | {task.upper()} | Epoch {epoch}")
            ax.set_xlabel("Target Expert", fontsize=16)
            if epoch == 1:
                ax.set_ylabel("Source Expert", fontsize=16)
            plt.tight_layout()
            
            # Save individual file
            save_name = f"matrix_{task}_epoch{epoch}.pdf"
            save_path = os.path.join(output_dir, save_name)
            plt.savefig(save_path)
            plt.close() # Close memory to prevent OOM loop
            
            print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, default="checkpoints/avg_C_matrix_*.pt", help="Glob pattern to find matrix files")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    plot_collaboration_matrices(args.glob, args.output_dir)
