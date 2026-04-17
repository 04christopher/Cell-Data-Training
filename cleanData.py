import scanpy as sc
import pandas as pd
import numpy as np

# 1. Load in 'backed' mode
file_path = "c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad"
adata = sc.read_h5ad(file_path, backed='r')

exclude_cells = ['RBC', 'Platelets', 'B_malignant', 'HSC_CD38neg', 'HSC_CD38pos',
                 'HSC_MK', 'HSC_erythroid', 'HSC_myeloid', 'HSC_prolif']

keep_indices = ~adata.obs['author_cell_type'].isin(exclude_cells)

all_eligible_indices = np.where(keep_indices)[0]
n_subsample = min(30000, len(all_eligible_indices))
chosen_indices = np.random.choice(all_eligible_indices, n_subsample, replace=False)

adata_subset = adata[chosen_indices].to_memory()

keep_metadata = ['donor_id', 'Status_on_day_collection_summary', 'author_cell_type', 'Site']
adata_subset.obs = adata_subset.obs[keep_metadata]

sc.pp.filter_genes(adata_subset, min_cells=10)
sc.pp.normalize_total(adata_subset, target_sum=1e4)
sc.pp.log1p(adata_subset)

#HVG Selection
sc.pp.highly_variable_genes(adata_subset, n_top_genes=2000, flavor='seurat', subset=True)

print(f"Final shape for ML: {adata_subset.shape}")
adata_subset.write_h5ad("stephenson_processed_for_ML.h5ad")