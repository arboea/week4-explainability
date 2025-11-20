# LIME explainability demo

This folder contains a small demo that trains a logistic regression model on the breast cancer dataset and uses LIME to explain predictions.

Files:

- `lime_explain.py`: trains model, produces a local explanation and a simple aggregated global feature-importance plot. Plots are saved to `explainability/outputs`.
- `requirements.txt`: Python packages needed to run the demo.

Quick start (PowerShell):

```powershell
cd c:\Users\andya\OneDrive\Documents\GitHub\week4-explainability\explainability
python -m pip install -r requirements.txt
python lime_explain.py
```

**Axes Key**

- **Local explanation plot (`outputs/lime_local_explanation.png`):**
	- **X-axis:** Contribution (LIME weight) — the signed weight from LIME's local linear surrogate for the chosen class. Positive values push the model's predicted probability toward the displayed class; negative values push it away. Magnitude indicates strength of the effect for that specific instance.
	- **Y-axis:** Feature rules — each entry shows a feature together with the local rule/threshold LIME used (e.g. `worst texture <= 25.22`). The bar for a rule shows that rule's contribution to the instance prediction.
	- **Note:** These weights are from a surrogate linear model fitted locally and are approximate; interpret them as directional, local influences on the predicted probability.

- **Global importance plot (`outputs/lime_global_importance.png`):**
	- **X-axis:** Average |LIME weight| — mean absolute weight across the sampled test instances (measures importance, not direction).
	- **Y-axis:** Feature names — features are sorted by aggregated importance (largest at the top of the plot).
	- **Note:** Aggregation uses absolute values, so the plot shows how strongly a feature influences predictions on average, but not whether it increases or decreases probability.

Outputs:

- `outputs/lime_local_explanation.png` — local LIME explanation for one test instance
- `outputs/lime_global_importance.png` — aggregated feature importance from several LIME explanations
