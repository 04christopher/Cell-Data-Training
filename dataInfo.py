import scanpy as sc

adata = sc.read_h5ad("c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad", backed='r')

print("--- Dataset Summary ---")
print(adata)

print("\n--- Cell Metadata Columns (obs) ---")
print(adata.obs.columns.tolist())

print("\n--- Gene Metadata Columns (var) ---")
print(adata.var.columns.tolist())