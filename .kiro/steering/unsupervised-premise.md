# Unsupervised Premise (Strict)

**In practice, only healthy data is available.** Damage labels exist in this dataset solely to *evaluate* the models, never to build them. Every detection pipeline in this project must remain strictly unsupervised.

## Rules

Anything that is fitted, learned, calibrated, or selected must use **only healthy samples (`Class 0`) from the training partition**. This includes:

- Model weights and parameters (OCSVM, Isolation Forest, Autoencoder, any new model).
- Preprocessing fitted on data: scalers, normalization statistics, feature filters (variance, correlation).
- Decision thresholds and decision rules (e.g., `mean + k * std` or a percentile of the healthy training error).
- Internal validation splits (e.g., early stopping) and any hyperparameter choice.

Damage information (`Class == 1`, `Dano_Percentual`, damaged files) may be used **only in the evaluation phase**:

- Fold metrics (accuracy, recall, confusion matrix), per-severity and per-EOV breakdowns.
- ROC curve / AUC as a diagnostic and for comparing models, never to pick a threshold, `k`, `nu`, `contamination`, or any other parameter.

## Do Not

- Calibrate thresholds or tune hyperparameters against damaged samples, test metrics, or AUC, even on a separate validation set.
- Propose "semi-supervised" shortcuts (e.g., a few labeled damaged samples) as a solution.
- Iterate model variants by maximizing test-fold metrics; changes must be motivated by healthy-only criteria or physical reasoning, with test metrics used only to report results.

If a proposed change would require damage labels before evaluation, flag it as a violation of this premise and do not implement it.

## Exception

The **Random Forest** pipeline is the explicitly supervised *upper bound baseline*. It exists only as a reference point for the unsupervised models and is not a candidate solution.
