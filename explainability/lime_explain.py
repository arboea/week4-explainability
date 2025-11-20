import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from lime.lime_tabular import LimeTabularExplainer


def train_model(X_train, y_train):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    return pipe


def explain_instance_with_lime(model, X_train, X_test, feature_names, class_names, out_dir):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
    )

    instance = X_test[0]
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=min(len(feature_names), 10),
    )

    print("Local explanation (top features and weights):")
    for feat, weight in exp.as_list():
        print(f"  {feat}: {weight:.4f}")

    fig = exp.as_pyplot_figure(label=1)
    plt.tight_layout()
    local_path = os.path.join(out_dir, "lime_local_explanation.png")
    fig.savefig(local_path)
    plt.close(fig)
    print(f"Saved local explanation plot to {local_path}")


def global_feature_importance_lime(model, X_train, X_test, feature_names, out_dir, sample_n=20):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        mode="classification",
    )

    n = min(sample_n, len(X_test))
    weights = {f: [] for f in feature_names}

    for i in range(n):
        exp = explainer.explain_instance(
            data_row=X_test[i],
            predict_fn=model.predict_proba,
            num_features=len(feature_names),
        )
        # exp.as_list(label) returns (feature, weight)
        for feat, w in exp.as_list(label=1):
            # feature string looks like 'mean area <= 800' or 'worst concavity > 0.2'
            # extract the feature name (before first space or before ' <=', ' >', ' >=', etc.)
            name = feat.split(' ')[0]
            if name in weights:
                weights[name].append(abs(w))

    avg_abs = {f: (np.mean(weights[f]) if len(weights[f]) > 0 else 0.0) for f in feature_names}
    sorted_feats = sorted(avg_abs.items(), key=lambda kv: kv[1], reverse=True)

    feats, vals = zip(*sorted_feats)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feats)), vals[::-1], align='center')
    plt.yticks(range(len(feats)), feats[::-1])
    plt.xlabel('Average |LIME weight|')
    plt.title('Global feature importance (aggregated LIME weights)')
    plt.tight_layout()
    global_path = os.path.join(out_dir, "lime_global_importance.png")
    plt.savefig(global_path)
    plt.close()
    print(f"Saved global importance plot to {global_path}")


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

    explain_instance_with_lime(model, X_train, X_test.values, list(X.columns), list(data.target_names), out_dir)
    global_feature_importance_lime(model, X_train, X_test.values, list(X.columns), out_dir, sample_n=20)


if __name__ == "__main__":
    main()
