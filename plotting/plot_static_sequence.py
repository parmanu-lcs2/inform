import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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


data = {
    "Task": ["MMLU", "MMLU", "HumanEval", "HumanEval", "GSM8K", "GSM8K"],
    "Configuration": ["Learned","Static"] * 3,
    "Performance": [82.7, 53, 88.4, 44, 72.75, 49] 
}

df = pd.DataFrame(data)
print(df)

plt.figure(figsize=(8, 4))

palette = {
    "Learned": "#1f77b4",
    "Static": "#ff7f0e" 
}

ax = sns.barplot(
    data=df,
    x="Task",
    y="Performance",
    hue="Configuration",
    palette=palette,
    dodge=True
)

hatches = ['/']*3 + ['*']*3

for patch, hatch in zip(ax.patches, hatches):
    patch.set_hatch(hatch)
    patch.set_edgecolor("black")
    patch.set_linewidth(1.2)

ax.set_xlabel("Task")
ax.set_ylabel("Performance")

ax.legend(
    title=None,
    frameon=False,
    loc="best",
)
plt.setp(ax.get_legend().get_texts(), fontsize='16')
plt.tight_layout()

out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "grouped_performance_comparison_static_sequence.pdf"))

plt.close()
