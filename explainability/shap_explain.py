import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import shap


def train_model(X_train, y_train):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    return pipe


def shap_explain(model, X_train, X_test, feature_names, out_dir):
    # Use SHAP's high-level Explainer for model-agnostic explanations
    # we pass model.predict_proba so SHAP explains predicted probabilities
    explainer = shap.Explainer(model.predict_proba, X_train)

    print("Computing SHAP values (this may take a moment)...")
    shap_values = explainer(X_test)

    # shap_values.values shape may be (n_samples, n_outputs, n_features)
    # for binary classification we pick the class 1 contributions
    if shap_values.values.ndim == 3:
        class_index = 1 if shap_values.values.shape[1] > 1 else 0
        vals = shap_values.values[:, class_index, :]
    else:
        vals = shap_values.values

    # Summary plot (global) using SHAP's summary_plot
    print("Rendering SHAP summary plot...")
    plt.figure(figsize=(8, 6))
    # pass the full shap_values object to summary_plot for best results
    shap.summary_plot(shap_values, X_test, show=False)
    summary_path = os.path.join(out_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"Saved SHAP summary plot to {summary_path}")

    # Simple aggregated global importance: mean |SHAP value|
    mean_abs = np.mean(np.abs(vals), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    feats_sorted = [feature_names[i] for i in sorted_idx]
    vals_sorted = mean_abs[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feats_sorted)), vals_sorted[::-1], align='center')
    plt.yticks(range(len(feats_sorted)), feats_sorted[::-1])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Global feature importance (SHAP)')
    plt.tight_layout()
    global_path = os.path.join(out_dir, "shap_global_importance.png")
    plt.savefig(global_path, dpi=150)
    plt.close()
    print(f"Saved SHAP global importance plot to {global_path}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    shap_explain(model, X_train, X_test, list(X.columns), out_dir)


if __name__ == "__main__":
    main()
