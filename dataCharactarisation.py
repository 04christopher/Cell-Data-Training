import joblib
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

adata = sc.read_h5ad("high_signal_covid_data.h5ad")
mapping = dict(zip(adata.var.index, adata.var['feature_name']))

#Define Model Groups
groups = [
    {
        "title": "",
        "filename": "linear_models_comparison.png",
        "models": {
            "Logistic Regression": "model_results/logistic_regression.joblib",
            "Linear SVM": "model_results/linear_svm.joblib"
        }
    },
    {
        "title": "",
        "filename": "ensemble_models_comparison.png",
        "models": {
            "Random Forest": "model_results/random_forest.joblib",
            "XGBoost": "model_results/xgboost.joblib"
        }
    }
]

for group in groups:
    fig, axes = plt.subplots(1, 2, figsize=(16, 10)) # Wide format for side-by-side

    for ax, (name, path) in zip(axes, group["models"].items()):
        try:
            model = joblib.load(path)

            if name in ["Logistic Regression", "Linear SVM"]:
                vals = model.coef_[0]
                label_x = "Model Coefficient (Weight)"
            else:
                vals = model.feature_importances_
                label_x = "Feature Importance"

            df = pd.DataFrame({
                'Symbol': [mapping.get(idx, idx) for idx in adata.var_names],
                'Value': vals,
                'Abs_Value': np.abs(vals)
            })

            top_20 = df.sort_values(by='Abs_Value', ascending=False).head(20)

            if name in ["Logistic Regression", "Linear SVM"]:
                colors = ['#C44E52' if x > 0 else '#4C72B0' for x in top_20['Value']]
                ax.axvline(0, color='black', linewidth=0.8, alpha=0.7)
            else:
                colors = '#55A868'

            ax.barh(top_20['Symbol'], top_20['Value'], color=colors)
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.set_xlabel(label_x, fontsize=12)
            ax.invert_yaxis()
            ax.grid(axis='x', linestyle='--', alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading {name}", ha='center')
            print(f"Error: {e}")

    plt.suptitle(group["title"], fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(group["filename"], dpi=300, bbox_inches='tight')
    plt.show()