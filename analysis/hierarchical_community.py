import sys
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from community import community_louvain


CHECKPOINT_PATH = sys.argv[1]

MODEL_NAMES = [
    'Model1', 'Model2', 'Model3', 'Model4', 'Model5',
    'Model6', 'Model7', 'Model8', 'Model9', 'Model10'
]


ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
state = ckpt["state_dict"]
config = ckpt["config"]

num = config["num_models"]
hidden = config["hidden_dim"]

W = state["C_head.weight"]
b = state["C_head.bias"]

W_mat = W.view(num, num, hidden)
b_mat = b.view(num, num)

for i in range(num):
    W_mat[i, i, :] = 0.0
    b_mat[i, i] = 0.0

edge_strength = torch.norm(W_mat, dim=2).numpy()
sym_matrix = (edge_strength + edge_strength.T) / 2


dist_matrix = np.max(sym_matrix) - sym_matrix
np.fill_diagonal(dist_matrix, 0.0)
condensed = squareform(dist_matrix)
Z = linkage(condensed, method='ward')

plt.figure(figsize=(8, 4))
dendrogram(Z, labels=MODEL_NAMES)
plt.title("Hierarchical Clustering of Models")
plt.tight_layout()
plt.savefig("model_clustering_dendrogram.pdf", dpi=300)
plt.close()


S = sym_matrix.copy()
threshold = np.percentile(S[S > 0], 75)

G = nx.Graph()
for name in MODEL_NAMES:
    G.add_node(name)

for i in range(num):
    for j in range(i + 1, num):
        w = float(S[i, j])
        if w >= threshold:
            G.add_edge(MODEL_NAMES[i], MODEL_NAMES[j], weight=w)

partition = community_louvain.best_partition(G, weight='weight')


plt.figure(figsize=(9, 7))

pos = nx.spring_layout(G, weight='weight', seed=42)

communities = set(partition.values())
colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))
color_map = {c: colors[i] for i, c in enumerate(communities)}

nx.draw_networkx_nodes(
    G, pos,
    node_color=[color_map[partition[n]] for n in G.nodes()],
    node_size=900
)

nx.draw_networkx_edges(G, pos, width=1.8)
nx.draw_networkx_labels(G, pos, font_size=10)


edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=8,
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none")
)

plt.title("Model Collaboration Communities (Louvain) with Edge Weights")
plt.axis("off")
plt.tight_layout()
plt.savefig("model_communities_with_weights.pdf", dpi=300)
plt.close()

print("Saved dendrogram: model_clustering_dendrogram.pdf")
print("Saved community graph: model_communities_with_weights.pdf")
