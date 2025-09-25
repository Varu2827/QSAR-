import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# ====== CONFIGURE ========
# =========================
@dataclass
class Config:
    csv_path: str = "merged_qsar_with_activity 1.csv"
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    run_svm: bool = True
    save_model_path: str = "qsar_rf_model.joblib"

CFG = Config()

# =========================
# ====== UTILITIES  =======
# =========================
def evaluate_cv(pipe, X, y, cv, name):
    scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1": "f1"}
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)
    print(f"\n=== {name} | {cv.get_n_splits()}-fold CV on TRAIN ===")
    for m in scoring.keys():
        print(f"{m:>9}: {res[f'test_{m}'].mean():.3f} ± {res[f'test_{m}'].std():.3f}")
    return res

def external_test_report(pipe, X_test, y_test, name="Model"):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"\n=== External Test ({name}) ===")
    print(f"Accuracy : {accuracy_score(y_test, pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.3f}")
    print(f"F1-score : {f1_score(y_test, pred):.3f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n", classification_report(y_test, pred, digits=3))

    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title(f"{name} ROC (External Test)")
    plt.show()

    PrecisionRecallDisplay.from_predictions(y_test, proba)
    plt.title(f"{name} PR (External Test)")
    plt.show()

# =========================
# ========= MAIN ==========
# =========================
def main():
    df = pd.read_csv("merged_qsar_with_activity_v1.csv")
    df['y'] = df['Activity']

    drop_cols = ["Ligand", "Ligand_ID", "File", "Activity"]
    df = df.drop(columns=drop_cols, errors="ignore")

    y = df['y']
    X = df.drop(columns=['y'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.test_size, stratify=y, random_state=CFG.random_state
    )

    numeric_features = X_train.columns.tolist()

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_split=2,
        class_weight="balanced", random_state=CFG.random_state, n_jobs=-1
    )

    pre_rf = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("varth", VarianceThreshold(threshold=0.0)),
        ]), numeric_features)
    ])

    pipe_rf = Pipeline([("pre", pre_rf), ("clf", rf)])

    if CFG.run_svm:
        svm = SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            C=1.0, gamma="scale", random_state=CFG.random_state
        )
        pre_svm = ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("varth", VarianceThreshold(threshold=0.0)),
                ("scale", StandardScaler())
            ]), numeric_features)
        ])
        pipe_svm = Pipeline([("pre", pre_svm), ("clf", svm)])

    # Cross-validation on training
    cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_state)
    cv_rf = evaluate_cv(pipe_rf, X_train, y_train, cv, "RandomForest")
    if CFG.run_svm:
        cv_svm = evaluate_cv(pipe_svm, X_train, y_train, cv, "SVM")

    # Train model and evaluate on test set
    pipe_rf.fit(X_train, y_train)
    external_test_report(pipe_rf, X_test, y_test, name="RandomForest")

    # Training Confusion Matrix
    train_pred = pipe_rf.predict(X_train)
    ConfusionMatrixDisplay.from_predictions(y_train, train_pred)
    plt.title("Confusion Matrix - Training Set")
    plt.show()

    if CFG.run_svm:
        pipe_svm.fit(X_train, y_train)
        external_test_report(pipe_svm, X_test, y_test, name="SVM")

    try:
        print("\nTop 30 RF feature importances (post-preprocessing):")
        pre = pipe_rf.named_steps["pre"]
        feat_names = pre.get_feature_names_out()
        rf_model = pipe_rf.named_steps["clf"]
        importances = pd.Series(rf_model.feature_importances_, index=feat_names).sort_values(ascending=False)
        print(importances.head(30))
    except Exception as e:
        print("Could not compute RF feature importances:", e)

    joblib.dump(pipe_rf, CFG.save_model_path)
    print(f"\n✅ Saved RF model to: {CFG.save_model_path}")

    # Plot class distribution
    compound_df = pd.read_csv(CFG.csv_path)
    sns.countplot(x="Activity", data=compound_df)
    plt.title("Compound Distribution: Activators vs Non-Activators")
    plt.xlabel("Class (0 = Non-Activator, 1 = Activator)")
    plt.ylabel("Number of Compounds")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
