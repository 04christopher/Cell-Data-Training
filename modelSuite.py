import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_curve, auc

os.makedirs("model_results", exist_ok=True)

#Load Data
adata = sc.read_h5ad("high_signal_covid_data.h5ad")
X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
y = adata.obs['target_score'].values
groups = adata.obs['donor_id'].values
feature_names = adata.var_names

# Pre-compute Topic Modeling
print("Extracting Gene Programs (LDA)...")
lda = LatentDirichletAllocation(n_components=10, random_state=42)
X_topics = lda.fit_transform(X)


models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, solver='saga', tol=0.1, class_weight='balanced'),
    "Linear SVM": LinearSVC(max_iter=5000, tol=0.1, class_weight='balanced', dual=False),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Topic-Based RF": RandomForestClassifier(n_estimators=100, n_jobs=-1)
}

gkf = GroupKFold(n_splits=5)
results_list = []
roc_data = {name: [] for name in models.keys()} # To store (y_true, y_probs) per fold

print(f"Benchmarking models across 5 donor-aware folds...")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_t_train, X_t_test = X_topics[train_idx], X_topics[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    for name, model in models.items():
        if name in ["Logistic Regression", "Linear SVM", "KNN"]:
            model.fit(X_train_s, y_train)
            y_probs = model.decision_function(X_test_s) if name == "Linear SVM" else model.predict_proba(X_test_s)[:, 1]
        elif name == "Topic-Based RF":
            model.fit(X_t_train, y_train)
            y_probs = model.predict_proba(X_t_test)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_probs = model.predict_proba(X_test)[:, 1]

        roc_data[name].append((y_test, y_probs))

        preds = (y_probs > 0.5) if name != "Linear SVM" else (y_probs > 0)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
        results_list.append({"Model": name, "Fold": fold, "F1-Score": f1, "Precision": prec, "Recall": rec})

    print(f"Completed Fold {fold+1}/5")

# Save Artifacts
df_res = pd.DataFrame(results_list)
df_res.to_csv("model_results/benchmark_metrics.csv", index=False)

np.save("model_results/X_test_s.npy", X_test_s)
np.save("model_results/X_test_raw.npy", X_test)
np.save("model_results/X_topics_test.npy", X_t_test)
np.save("model_results/y_test.npy", y_test)

for name, model in models.items():
    joblib.dump(model, f"model_results/{name.replace(' ', '_').lower()}.joblib")

joblib.dump(scaler, "model_results/scaler.joblib")
joblib.dump(lda, "model_results/lda_transformer.joblib")

#oerformance
plt.figure(figsize=(10, 6))
sns.barplot(data=df_res, x="Model", y="F1-Score", hue="Model", palette="viridis", legend=False)
plt.title("Model Performance: Healthy vs. Severe (Donor-Aware CV)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_results/model_comparison.png")

#roc curves
plt.figure(figsize=(10, 8))
mean_fpr = np.linspace(0, 1, 100)

for name in models.keys():
    tprs = []
    aucs = []

    for y_true, y_score in roc_data[name]:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, label=f'{name} (Mean AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Aggregate 5-Fold ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("model_results/aggregate_roc_comparison.png")


summary = df_res.groupby("Model")[["F1-Score", "Precision", "Recall"]].mean()
print("\n--- FINAL BENCHMARK SUMMARY ---")
print(summary.sort_values("F1-Score", ascending=False))
print("\nAll files and plots saved to 'model_results' directory.")