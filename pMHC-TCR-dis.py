import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files
mhc_file = "MHC_antigen_test.csv"  # Replace with your file path
tcr_file = "TCR_test_pos.csv"      # Replace with your file path

mhc_antigen_data = pd.read_csv(mhc_file)
tcr_test_data = pd.read_csv(tcr_file)

# Set index to the ID column
mhc_antigen_data.set_index('Unnamed: 0', inplace=True)
tcr_test_data.set_index('Unnamed: 0', inplace=True)

# Ensure both datasets have the same samples
aligned_data = mhc_antigen_data.join(tcr_test_data, lsuffix='_mhc', rsuffix='_tcr', how='inner')

# Extract features for both models
mhc_features = aligned_data.filter(like='_mhc').values
tcr_features = aligned_data.filter(like='_tcr').values

# Perform t-SNE on the features
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
mhc_tsne = tsne.fit_transform(mhc_features)
tcr_tsne = tsne.fit_transform(tcr_features)

# Plot with density regions, scatter points, and contours using t-SNE results
plt.figure(figsize=(12, 8))
#plt.figure()

# Model 1 (MHC) density
sns.kdeplot(
    x=mhc_tsne[:, 0], 
    y=mhc_tsne[:, 1], 
    fill=True, 
    alpha=0.5,  # Reduce transparency
    levels=5,  # Fewer levels for simpler contour
    cmap='Oranges', 
    #label='Model 1 (MHC)'
)

# Model 2 (TCR) density
sns.kdeplot(
    x=tcr_tsne[:, 0], 
    y=tcr_tsne[:, 1], 
    fill=True, 
    alpha=0.5,  # Reduce transparency
    levels=5,  # Fewer levels for simpler contour
    cmap='Blues', 
    #label='Model 2 (TCR)'
)

# Scatter points for Model 1 (MHC)
#plt.scatter(mhc_tsne[:, 0], mhc_tsne[:, 1], alpha=0.2, color='darkorange', marker='o', s=10, label='neMHCpan output')
plt.scatter(mhc_tsne[:, 0], mhc_tsne[:, 1], alpha=0.2, color='#ff7f0e', marker='o', s=10, label='neMHCpan output')
plt.gca().set_aspect('equal', adjustable='box')

# Scatter points for Model 2 (TCR)
#plt.scatter(tcr_tsne[:, 0], tcr_tsne[:, 1], alpha=0.2, color='darkgreen', marker='o', s=10, label='TCR VAE output')
plt.scatter(tcr_tsne[:, 0], tcr_tsne[:, 1], alpha=0.2, color='#1f77b4', marker='o', s=10, label='TCR VAE output')
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('t-SNE Component 1', fontsize=14, fontweight='bold')
plt.ylabel('t-SNE Component 2', fontsize=14, fontweight='bold')
#plt.title('2D t-SNE Distribution with Points and Contours')
plt.legend(markerscale=2.0, fontsize=14)

# Adjust font size for tick labels
#plt.xticks(fontsize=14, fontweight='bold')  # x-axis tick labels
#plt.yticks(fontsize=14, fontweight='bold')  # y-axis tick labels


# Save the figure in multiple formats
output_filename = "tsne_visualization"  # Customize the base filename as needed
#formats = ['pdf', 'svg', 'jpg', 'png', 'eps']  # Formats to save
formats = ['pdf']  # Formats to save
for fmt in formats:
    plt.savefig(f"{output_filename}.{fmt}", format=fmt, transparent=True)
    print(f"Saved figure as {output_filename}.{fmt}")

plt.show()