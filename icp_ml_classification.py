"""
ICP Machine Learning Classification Pipeline
Author: Halis Karaveli
Year: 2025

This script performs classification of intracranial pressure (ICP) status using clinical and gene expression
data from the NASA GeneLab OSD-364 dataset. It includes data preprocessing, supervised learning (Logistic Regression,
Random Forest, KNN), model evaluation, SHAP explainability, gene-level statistical analysis, and clustering.

Instructions:
- Update the file paths below to point to your own data files.
- Install the required Python packages: pandas, numpy, seaborn, matplotlib, scikit-learn, shap
    pip install pandas numpy seaborn matplotlib scikit-learn shap
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, RocCurveDisplay
)
import shap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pandas.api.types import is_numeric_dtype

# -------------------------------
# 0. FIX WINDOWS MKL BUG (Optional, for some Windows users)
# -------------------------------
os.environ["OMP_NUM_THREADS"] = "1"

# -------------------------------
# 1. LOAD DATA
# -------------------------------
# Update these paths to your local files:
sample_path = "YOUR_PATH_HERE/s_OSD-364.txt"
assay_path = "YOUR_PATH_HERE/a_OSD-364_transcription-profiling_real-time-pcr_QuantStudio_12K_Flex.txt"

samples = pd.read_csv(sample_path, sep="\t", dtype=str).reset_index(drop=True)
assay = pd.read_csv(assay_path, sep="\t", dtype=str).reset_index(drop=True)

samples = samples.loc[:, ~samples.columns.duplicated()].copy()
assay = assay.loc[:, ~assay.columns.duplicated()].copy()

# Convert gene columns to numeric
for col in assay.columns:
    if col != 'Sample Name':
        assay[col] = pd.to_numeric(assay[col], errors='coerce')

# Merge clinical and assay data
merged = samples.merge(assay, on='Sample Name', how='left')

# Convert clinical columns to numeric
clinical_columns = ['Characteristics[Weight]', 'Characteristics[Height]', 'Characteristics[Age]', 'Parameter Value[Average ICP]']
for col in clinical_columns:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')

# Feature engineering: BMI and ICP group
merged['BMI'] = merged['Characteristics[Weight]'] / (merged['Characteristics[Height]'] ** 2)
merged['ICP_Group'] = merged['Factor Value[Intracranial Pressure]'].map({
    'high ICP': 1,
    'normal to mild ICP': 0
})

# Drop columns/rows with all missing data, fill missing numerics with median
merged = merged.dropna(axis=1, how='all')
numeric_cols = merged.select_dtypes(include='number').columns
merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].median())
merged = merged.dropna(subset=['ICP_Group'])

# -------------------------------
# 2. SUPERVISED MODELING
# -------------------------------
features = ['Characteristics[Age]', 'Characteristics[Height]', 'Characteristics[Weight]', 'BMI']
X = merged[features]
y = merged['ICP_Group']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Train model, print classification metrics, and plot confusion matrix & ROC curve.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"{name} - ROC Curve")
    plt.tight_layout()
    plt.show()

# Evaluate each model
for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)

# -------------------------------
# 3. CROSS-VALIDATION SUMMARY
# -------------------------------
cv_results = {}
for name, model in models.items():
    scores = cross_validate(model, X_scaled, y, cv=3, scoring=["accuracy", "precision", "recall", "roc_auc"])
    cv_results[name] = pd.DataFrame(scores).mean()

cv_summary = pd.DataFrame(cv_results).T
print("\nCross-Validation Summary:")
print(cv_summary)

plt.figure(figsize=(8, 4))
sns.heatmap(cv_summary, annot=True, cmap='YlGnBu', fmt=".3f")
plt.title("Cross-Validation Metrics")
plt.tight_layout()
plt.show()

# -------------------------------
# 4. HYPERPARAMETER TUNING (RANDOM FOREST)
# -------------------------------
param_grid_rf = {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='roc_auc')
rf_grid.fit(X_scaled, y)
print("\nBest Random Forest Params:", rf_grid.best_params_)

# -------------------------------
# 5. MODEL EXPLAINABILITY (SHAP)
# -------------------------------
explainer = shap.Explainer(rf_grid.best_estimator_)
shap_values = explainer(X_scaled)
shap.summary_plot(shap_values, features=X, feature_names=features)

# -------------------------------
# 6. FEATURE IMPORTANCE (RANDOM FOREST)
# -------------------------------
def plot_importance(model, feature_names, num=10):
    """
    Plot feature importance for the Random Forest model.
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(num)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=importance_df,
        x='Importance',
        y='Feature',
        hue='Feature',
        palette=sns.color_palette("viridis", n_colors=importance_df.shape[0]),
        dodge=False,
        legend=False
    )
    plt.title("Feature Importance (Random Forest)", fontsize=14)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()

plot_importance(rf_grid.best_estimator_, features)

# -------------------------------
# 7. GENE-LEVEL DIFFERENTIAL EXPRESSION (MANN-WHITNEY U)
# -------------------------------
gene_cols = [col for col in assay.columns if col != 'Sample Name']
valid_genes = [
    gene for gene in gene_cols
    if gene in merged.columns and
       is_numeric_dtype(merged[gene]) and
       merged[gene].nunique() > 1
]

results = []
for gene in valid_genes:
    data = merged[[gene, 'ICP_Group']].dropna()
    group0 = data[data['ICP_Group'] == 0][gene]
    group1 = data[data['ICP_Group'] == 1][gene]
    if len(group0) > 0 and len(group1) > 0:
        stat, pval = mannwhitneyu(group0, group1)
        results.append({'Gene': gene, 'p-value': pval})

results_df = pd.DataFrame(results)
results_df['-log10(p-value)'] = -np.log10(results_df['p-value'])
top_genes_df = results_df.sort_values(by='-log10(p-value)', ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_genes_df,
    x='Gene',
    y='-log10(p-value)',
    hue='Gene',
    palette='magma',
    dodge=False,
    legend=False
)
plt.xticks(rotation=90)
plt.title("Top Differentially Expressed Genes (ICP Groups)")
plt.ylabel("-log10(p-value)")
plt.xlabel("Gene")
plt.tight_layout()
plt.show()

# -------------------------------
# 8. PCA & K-MEANS CLUSTERING
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(assay.drop(columns=['Sample Name']).fillna(0))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=merged['ICP_Group'])
plt.title("PCA of Gene Expression by ICP Group")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1')
plt.title("KMeans Clustering on PCA-Reduced Gene Expression")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
