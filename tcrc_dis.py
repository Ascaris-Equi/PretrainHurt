import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns  # Optional, for more aesthetically pleasing plots

# 1. Read CSV files, assuming the first column contains TCR sequences
df1 = pd.read_csv('all_cdr3_aa_beta_processed.csv', header=0)
df2 = pd.read_csv('TCR_10k_bg_seq.csv', header=0)

# Add a "label" column to differentiate datasets
df1['label'] = 'OTS 1M dataset'
df2['label'] = 'TCR 10k dataset'

# 2. Extract sequences and labels separately
sequences_all = df1.iloc[:, 0].astype(str)  # TCR sequences from df1
labels_all = df1['label'].values

sequences_pretrain = df2.iloc[:, 0].astype(str)  # TCR sequences from df2
labels_pretrain = df2['label'].values

# 3. Feature Engineering: One-Hot encode the first 20 amino acids
AAS = list("ACDEFGHIKLMNPQRSTVWY")  # Common 20 amino acids
aa_to_idx = {aa: i for i, aa in enumerate(AAS)}

max_len = 20

def one_hot_encode(sequences, max_len, aa_to_idx):
    features = []
    for seq in sequences:
        seq = seq[:max_len].ljust(max_len, '-')  # Truncate or pad with '-'
        one_hot_vec = np.zeros((max_len, len(AAS)), dtype=np.float32)
        for i, aa in enumerate(seq):
            if aa in aa_to_idx:
                one_hot_vec[i, aa_to_idx[aa]] = 1
        features.append(one_hot_vec.flatten())
    return np.array(features)

features_all = one_hot_encode(sequences_all, max_len, aa_to_idx)
features_pretrain = one_hot_encode(sequences_pretrain, max_len, aa_to_idx)

# 4. Standardize Features
# Fit the scaler on the pretrain data and transform both datasets
scaler = StandardScaler()
features_pretrain_scaled = scaler.fit_transform(features_pretrain)
features_all_scaled = scaler.transform(features_all)

# 5. Perform PCA
# Fit PCA on the pretrain data
pca = PCA(n_components=2)
pca_pretrain = pca.fit_transform(features_pretrain_scaled)

# Transform the ALL data using the fitted PCA
pca_all = pca.transform(features_all_scaled)

# Check PCA explained variance
explained_variance = pca.explained_variance_ratio_
print(f"PCA explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")

# 6. Prepare DataFrame for Plotting
df_pretrain_pca = pd.DataFrame({
    'pca_1': pca_pretrain[:, 0],
    'pca_2': pca_pretrain[:, 1],
    'label': labels_pretrain
})

df_all_pca = pd.DataFrame({
    'pca_1': pca_all[:, 0],
    'pca_2': pca_all[:, 1],
    'label': labels_all
})

# Combine for plotting
df_plot = pd.concat([df_pretrain_pca, df_all_pca], ignore_index=True)
df_plot['pca_2'] = -df_plot['pca_2']

# 7. Plotting
#plt.figure(figsize=(10, 8))
plt.figure(figsize=(12, 8))
sns.set(style="white")  # Set style to 'white' to remove grid lines

# Define color mapping for Nature style: Blue and Orange
color_map = {
    'OTS 1M dataset': '#1f77b4',  # Nature blue
    'TCR 10k dataset': '#ff7f0e'  # Nature orange
}

# Define point size
point_size = 10  # Adjust as needed

# Plot each group separately
for group, color in color_map.items():
    subset = df_plot[df_plot['label'] == group]
    plt.scatter(
        subset['pca_1'],
        subset['pca_2'],
        c=color,
        label=group,
        alpha=0.6,
        edgecolors='w',
        linewidths=0,
        s=point_size,
        rasterized=True
    )

#plt.title("PCA of TCR Sequences (Pretrain Basis)", fontsize=16)
#plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)", fontsize=14, fontweight='bold')
#plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)", fontsize=14, fontweight='bold')
plt.xlabel('PC1 (1.7%)', fontsize=14, fontweight='bold')
plt.ylabel('PC2 (1.5%)', fontsize=14, fontweight='bold')
#plt.legend(title='Label')
plt.legend()
plt.tight_layout()
plt.grid(False)

# Adjust font size for tick labels
#plt.xticks(fontsize=14, fontweight='bold')  # x-axis tick labels
#plt.yticks(fontsize=14, fontweight='bold')  # y-axis tick labels


plt.legend(markerscale=2.0, fontsize=14)

# 关键：设置坐标轴的横纵比为相同
plt.gca().set_aspect('equal', adjustable='box')

# Save the plot in multiple formats
output_filename = 'PCA_of_TCR_Sequences'  # Without file extension
# formats = ['jpg']  # Add 'eps' if needed
formats = ['pdf','eps']  # Add 'eps' if needed

for fmt in formats:
    plt.savefig(f"{output_filename}.{fmt}", format=fmt, transparent=True)
    print(f"Saved figure as {output_filename}.{fmt}")

plt.show()