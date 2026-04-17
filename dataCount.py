import scanpy as sc

adata = sc.read_h5ad("c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad", backed='r')

cell_types = sorted(adata.obs['author_cell_type'].cat.categories.tolist())

print("--- COPY AND PASTE THE LIST BELOW ---")
print(cell_types)