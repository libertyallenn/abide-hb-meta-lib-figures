import os
import os.path as op
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mscolors
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import pdist 


data_dir = "./data"
out_dir = "./figures/hierarchical_dendrogram"
fig_save_dir = out_dir
os.makedirs(fig_save_dir, exist_ok=True)
data_path = "./data/clustering_data_matrix.npy"
data_matrix = np.load(data_path)

Xc = data_matrix - data_matrix.mean(axis=1, keepdims=True)
n = np.linalg.norm(Xc, axis=1, keepdims=True)
n[n == 0] = 1.0
Xcorr = Xc / n
metric = "euclidean"
method = "ward"
condensed = pdist(Xcorr, metric=metric)
linkage_matrix = linkage(condensed, method=method)
distances = linkage_matrix[:, 2]
n = linkage_matrix.shape[0] + 1  

def threshold_for_k(Z, k):
    n = Z.shape[0] + 1
    lo_idx = n - k - 1
    hi_idx = n - k
    lo = Z[lo_idx, 2] if lo_idx >= 0 else 0.0
    hi = Z[hi_idx, 2]
    t = (lo + hi) / 2.0 if hi != lo else hi * 0.9999
    return t
thresholds = {k: threshold_for_k(linkage_matrix, k) for k in range(2, 10)}
merge_heights = {k: linkage_matrix[linkage_matrix.shape[0] - k, 2] for k in range (2, 10)}
jumps = {k: merge_heights[k] - merge_heights[k+1] for k in range (2, 9)}
best_k = max(jumps, key=jumps.get)
best_cutoff = thresholds[best_k]

fig, ax = plt.subplots(figsize=(10, 8)) # w, h
husl_colors = sns.color_palette("husl", 8)
custom_colors = [
    husl_colors[7],
    husl_colors[4],
    husl_colors[3],
]
husl_hex = [mscolors.to_hex(c) for c in custom_colors]
cluster_labels = fcluster(linkage_matrix, t=best_k, criterion="maxclust")
cluster_colors = [husl_hex[label-1] for label in cluster_labels]
set_link_color_palette(husl_hex)

def link_color_func(link):
    if link < len(cluster_labels):
        return cluster_colors[link]
    else:
        left = int(linkage_matrix[link - len(cluster_labels), 0])
        right = int(linkage_matrix[link - len(cluster_labels), 1])
        left_color = link_color_func(left)
        right_color = link_color_func(right)
        return left_color if left_color == right_color else "gray"
dendrogram(
    linkage_matrix,
    ax=ax,
    color_threshold=np.inf,
    link_color_func=link_color_func,
    above_threshold_color="gray",
    no_labels=True,
    leaf_font_size=10
)
ax.axhline(
    y=best_cutoff,
    color="red", 
    linestyle='--',
    linewidth=2,
    alpha=0.8
)
for label in ax.get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontsize(10)

legend_font = font_manager.FontProperties(family="Times New Roman", size=8)
delta = "\u0394"
legend = [
    f"Best k = {best_k}",
    f"Threshold (midpoint) = {thresholds[best_k]:.4f}",
    f"Merge Linkage Distance = {merge_heights[best_k]:.4f}", 
    f"{delta} Linkage Distance = {jumps[best_k]:.4f}"]
empty_handles = [Line2D([0], [0], color='none') for _ in legend]
ax.legend(handles=empty_handles, labels=legend, loc='upper right', prop=legend_font, 
          frameon=True, framealpha=0.9, handlelength=0, handletextpad=0, borderaxespad=0.5, borderpad=0.7, labelspacing=0.1)
ax.set_xlabel("Sample index (ASD participants)", fontname="Times New Roman", fontsize=12)
ax.set_ylabel("rsFC dissimilarity (linkage distance)", fontname="Times New Roman", fontsize=12) 
fig.suptitle("Hierarchical clustering of habenula rsFC patterns (k = 2-9)", x=0.5, y=0.95, fontname="Times New Roman", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(wspace=0.2)
out_path = op.join(fig_save_dir, "hierarchical_dendrogram.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

delta = "\u0394"
df = pd.DataFrame({
    "Number of clusters (k)": list(range(2,10)),
    "Threshold (midpoint)": [thresholds[k] for k in range (2,10)],
    "Merge Linkage Distance": [merge_heights[k] for k in range (2,10)],
    f"{delta} Linkage Distance": [jumps.get(k, np.nan) for k in range (2,10)]
})
print(df.to_string(index=False))
csv_path = op.join(fig_save_dir, "best_k_value.csv")
df.to_csv(csv_path, index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("tight")
ax.axis("off")
header_colors = ["#0000001A"]*len(df.columns)
row_colors = [["#fffefeff"]*len(df.columns) if i%2==0 else ["#0000001A"]*len(df.columns) for i in range(len(df))]
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    colColours=header_colors,
    cellColours=row_colors,
    cellLoc="center",
    loc="center"
)
table.auto_set_font_size(False)
table.scale(1, 2.5)
for col in range(len(df.columns)):
    cell = table[0, col]
    cell.set_text_props(fontweight="bold", fontname="Times New Roman", fontsize=12, wrap=True)
for row in range(1, len(df) + 1):
    for col in range(len(df.columns)):
        cell = table[row, col]
        cell.set_text_props(fontname="Times New Roman", fontsize=11, wrap=False)
ax.set_title("Hierarchical Cluster Solutions and Linkage Distance Metrics", fontname="Times New Roman", fontsize=13, pad=10) 
table_path = op.join(fig_save_dir, "best_k_value_table_fig.png")
plt.savefig(table_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
