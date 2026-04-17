import joblib
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

model = joblib.load("model_results/logistic_regression.joblib")
adata = sc.read_h5ad("high_signal_covid_data.h5ad")

feature_names = adata.var_names

# Get the coefficients
coeffs = model.coef_[0]

gene_importance = pd.DataFrame({
    'Gene': feature_names,
    'Importance': coeffs,
    'Absolute_Importance': np.abs(coeffs)
})

#Sort by absolute importance
top_20_genes = gene_importance.sort_values(by='Absolute_Importance', ascending=False).head(20)
# Mapping Ensembl IDs to Symbols using the adata.var table
id_to_symbol = adata.var['gene_ids-0'].reset_index() # Adjust column names if needed
mapping = dict(zip(adata.var['gene_ids-0'], adata.var_names))
gene_importance['Symbol'] = gene_importance['Gene'].map(mapping)

plt.barh(top_20_genes['Symbol'], top_20_genes['Importance'])
print("--- TOP 20 PREDICTIVE GENES ---")
print(top_20_genes[['Gene', 'Importance']])
#plot
plt.figure(figsize=(10, 8))
plt.barh(top_20_genes['Gene'], top_20_genes['Importance'], color=['red' if x > 0 else 'blue' for x in top_20_genes['Importance']])
plt.title("Gene Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value (Positive = Severe, Negative = Healthy)")
plt.gca().invert_yaxis()
plt.show()