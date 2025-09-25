import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler

# --------------------------
# INPUT SECTION
# --------------------------
# Replace these with your actual data
y_test = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])  # True labels (example)
y_prob_rf = np.random.rand(len(y_test))  # RF predicted probabilities (example)
y_prob_svm = np.random.rand(len(y_test))  # SVM predicted probabilities (example)

# Descriptor matrix for PCA & Williams plot (replace with your actual descriptor data)
# Example: descriptors = pd.read_csv("descriptors.csv")
descriptors = pd.DataFrame({
    "MolWt": np.random.rand(10)*300,
    "LogP": np.random.rand(10)*5,
    "HBA": np.random.randint(0, 10, 10),
    "HBD": np.random.randint(0, 5, 10),
    "TPSA": np.random.rand(10)*100
})

# Predicted labels (0/1) for residual calculation
y_pred_rf = (y_prob_rf > 0.5).astype(int)
residuals = y_test - y_pred_rf

# --------------------------
# 1. PRECISION–RECALL CURVES
# --------------------------
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
ap_rf = average_precision_score(y_test, y_prob_rf)

prec_svm, rec_svm, _ = precision_recall_curve(y_test, y_prob_svm)
ap_svm = average_precision_score(y_test, y_prob_svm)

plt.figure(figsize=(6, 5))
plt.plot(rec_rf, prec_rf, color='blue', label=f'RF (AP={ap_rf:.2f})')
plt.plot(rec_svm, prec_svm, color='green', label=f'SVM (AP={ap_svm:.2f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves (External Test)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("PR_Curves.png", dpi=300)
plt.close()

print("[INFO] Precision-Recall curves saved as PR_Curves.png")

# --------------------------
# 2. WILLIAMS PLOT (Applicability Domain)
# --------------------------
# Standardize descriptors
X = StandardScaler().fit_transform(descriptors)

# Calculate leverage
H = X @ np.linalg.inv(X.T @ X) @ X.T
leverage = np.diag(H)
h_star = 3 * (X.shape[1] + 1) / X.shape[0]  # Threshold

# Ensure residuals match size
if len(residuals) != len(leverage):
    residuals = np.resize(residuals, len(leverage))

plt.figure(figsize=(6, 5))
plt.scatter(leverage, residuals, c=y_test, cmap='coolwarm', edgecolor='k')
plt.axhline(y=3, color='red', linestyle='--', label='±3 Residual')
plt.axhline(y=-3, color='red', linestyle='--')
plt.axvline(x=h_star, color='purple', linestyle='--', label=f'h*={h_star:.2f}')
plt.xlabel("Leverage (h)")
plt.ylabel("Standardized Residuals")
plt.title("Williams Plot (Applicability Domain)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Williams_Plot.png", dpi=300)
plt.close()

print("[INFO] Williams plot saved as Williams_Plot.png")

# --------------------------
# 3. PCA PLOT
# --------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
expl_var = pca.explained_variance_ratio_

plt.figure(figsize=(6, 5))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=y_test, palette=['blue', 'orange'], s=80)
plt.xlabel(f"PC1 ({expl_var[0]*100:.1f}% Var)")
plt.ylabel(f"PC2 ({expl_var[1]*100:.1f}% Var)")
plt.title("PCA of Chemical Descriptors")
plt.legend(title="Class", labels=["Non-Activator", "Activator"])
plt.tight_layout()
plt.savefig("PCA_Plot.png", dpi=300)
plt.close()

print("[INFO] PCA plot saved as PCA_Plot.png")

