import scanpy as sc
adata = sc.read_h5ad("stephenson_processed_for_ML.h5ad")
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['Status_on_day_collection_summary', 'author_cell_type'])