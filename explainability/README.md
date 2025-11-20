# LIME explainability demo
REFLECTION

This exercise highlights the need to maintain explainability to a wider audience. The resulting graphic from the AI's work does not adequetly explain on its own what the data is showing. The simple labels on the X and Y axis do not convey the message in basic and easily understandable terms. Human intervention is required to clearly articulate the findings. 

My hope would be that over time, the output can be improved to provide much better explainability without impacting accuracy of the output. I believe that with the proper guidelines and safeguards in place, this can be accomplished. AI as it exists currently cannot be allowed to simply trusted without oversight and some form of vetting process. While AI is a powerful tool, it does not yet know how to provide its output in a manner that is easy for many to digest and does not provide the context behind the output unless directed to do so. This puts a greater emphasis on expainability over accuracy. 

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

SHAP

- `shap_explain.py`: trains the same logistic regression model and produces SHAP visualizations.
- `outputs/shap_summary.png` — SHAP summary plot (global view of feature impacts)
- `outputs/shap_global_importance.png` — aggregated mean |SHAP value| bar plot

To run the SHAP demo (PowerShell):

```powershell
cd c:\Users\andya\OneDrive\Documents\GitHub\week4-explainability\explainability
python -m pip install -r requirements.txt
python shap_explain.py
```

**Axes Key (SHAP)**

- **SHAP summary plot (`outputs/shap_summary.png`):**
	- **X-axis:** SHAP value — impact on model output (predicted probability for the explained class). Negative values push prediction toward class 0, positive toward class 1. Each point is one sample-feature contribution; color encodes the raw feature value (low→high).
	- **Y-axis:** Feature names — features sorted by importance (top = most important by mean absolute SHAP value).

- **SHAP global importance (`outputs/shap_global_importance.png`):**
	- **X-axis:** Mean |SHAP value| — average absolute impact across samples (importance magnitude).
	- **Y-axis:** Feature names — sorted by mean absolute SHAP value.

