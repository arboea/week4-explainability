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

Outputs:

- `outputs/lime_local_explanation.png` — local LIME explanation for one test instance
- `outputs/lime_global_importance.png` — aggregated feature importance from several LIME explanations
