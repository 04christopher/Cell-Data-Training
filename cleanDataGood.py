import scanpy as sc
import numpy as np
import pandas as pd

# Load data in backed mode to save RAM
file_path = "c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad"
adata = sc.read_h5ad(file_path, backed='r')

# Define High-Signal Comparison (Binary)
keep_status = ['Healthy', 'Severe', 'Critical']
cells_to_keep = adata.obs['Status_on_day_collection_summary'].isin(keep_status)

# Donor-Balanced Subsampling
all_indices = np.where(cells_to_keep)[0]
obs_subset = adata.obs.iloc[all_indices]

balanced_indices = (
    obs_subset.groupby('donor_id')
    .apply(lambda x: x.sample(n=min(len(x), 500), random_state=42))
    .index.get_level_values(1)
)

#Bring into Memory and Label
adata_final = adata[balanced_indices].to_memory()

# Create a clean binary label: 0 for Healthy, 1 for Severe/Critical
adata_final.obs['target_score'] = adata_final.obs['Status_on_day_collection_summary'].map({
    'Healthy': 0,
    'Severe': 1,
    'Critical': 1
})

sc.pp.filter_genes(adata_final, min_cells=50) # Remove rare/noisy genes
sc.pp.normalize_total(adata_final, target_sum=1e4)
sc.pp.log1p(adata_final)

# HVG
sc.pp.highly_variable_genes(adata_final, n_top_genes=2000, flavor='seurat', subset=True)

#
print(f"Final Dataset: {adata_final.n_obs} cells and {adata_final.n_vars} genes.")
adata_final.write_h5ad("high_signal_covid_data.h5ad")