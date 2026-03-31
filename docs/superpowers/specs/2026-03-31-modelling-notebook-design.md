# Modelling Notebook Design

## Goal

Complete `notebooks/modelling.ipynb` so it produces a reviewer-friendly loss-prediction baseline for `is_loss_making`, compares multiple classification models with PyCaret including `lightgbm`, and saves a self-contained model artifact to `app/artifacts/model.pkl`.

## Scope

This spec only covers Part 1 of the take-home inside the modelling notebook. It does not define the downstream app, tool, or evaluation changes.

## Design Principles

- Prefer a reviewer-friendly baseline over pure benchmark chasing.
- Use PyCaret for structured model comparison, but narrow the candidate set to keep runtime and reasoning tractable.
- Allow a tree-based final model if it clearly outperforms simpler options and the notebook explains its drivers and limitations clearly.
- Keep notebook analysis concise and decision-oriented rather than producing exhaustive exploratory output.
- Reuse existing utilities from `utils/` for configuration loading and artifact persistence where that reuse is practical and improves consistency.

## Inputs And Dependencies

- `README.md` provides the modelling objective and submission expectations.
- `data/records.csv` is the historical labelled dataset.
- `conf/classification.toml` provides training settings and the feature list.
- `utils/config/config.py` and related config helpers provide reusable config loading.
- `utils/ml/pickle_handler.py` can be reused to persist the trained artifact.
- `pycaret.classification` is used for setup, comparison, finalization, and prediction.

## Notebook Structure

### 1. Bootstrap And Configuration

The notebook should:

- Ensure relative paths work when run interactively or through `nbconvert`.
- Load configuration from `conf/classification.toml`.
- Define shared constants such as the target column, feature list, and artifact path.

### 2. Data Exploration

The TODO exploration cells should cover:

- Dataset shape, dtypes, and missing-value counts.
- Class balance for `is_loss_making`.
- Numeric distributions for `limit`, `premium`, `prior_claims`, `years_trading`, and optionally `loss_ratio` for understanding the label.
- Categorical loss-rate summaries across `risk_type`, `territory`, `industry`, and `broker`.
- Numeric correlation inspection focused on the target and obvious feature relationships.

The exploration should highlight the known data-quality issues in `records.csv`, especially:

- Missing values
- Categorical noise such as inconsistent casing or whitespace
- Impossible or suspicious numeric values
- Leakage risk from `loss_ratio`

### 3. Feature Engineering

The feature-engineering cells should:

- Remove target leakage by excluding `loss_ratio` from modelling features.
- Normalize categorical text fields using lightweight cleaning such as trimming whitespace and lowercasing.
- Handle missing values conservatively so the training pipeline remains robust.
- Add simple reviewer-meaningful derived features, with `premium_rate = premium / limit` as the primary derived feature.
- Keep derived features limited to those that are easy to explain to a reviewer.

The final modelling dataset should remain compact and interpretable.

### 4. Model Comparison With PyCaret

The training section should:

- Use `pycaret.classification.setup()` on the engineered dataset.
- Compare a constrained set of classifiers rather than every available estimator.
- Explicitly include `lightgbm` in the candidate set.
- Sort by `AUC` while still reviewing the broader trade-off between discrimination and reviewer usefulness.

Recommended candidate families:

- Logistic regression
- Decision tree
- Random forest or extra trees
- LightGBM

The notebook should not blindly save the top scorer without commentary. It should show judgment about why the selected model is acceptable for reviewer support.

### 5. Evaluation And Explainability

The evaluation cells should:

- Produce holdout predictions in a way that works with the finalized PyCaret model.
- Show compact metrics such as classification report, confusion matrix, and ROC-AUC if practical within the notebook flow.
- Provide a short explanation of what the model appears to rely on.

Explainability expectations:

- If the selected model is linear, show coefficients.
- If the selected model is tree-based, show feature importances.
- The notebook should explicitly state that this is a reviewer-support model, not a fully autonomous decision system.

### 6. Artifact Saving

The saved artifact should be self-contained enough for app loading. It should include:

- The finalized model or pipeline
- The feature column list
- Relevant configuration metadata

The artifact should be written to `app/artifacts/model.pkl`.

If reuse is practical, the notebook should use `utils/ml/pickle_handler.py` for persistence instead of ad hoc pickle-writing logic.

## Model-Selection Standard

The final selection should optimize for:

1. Reasonable predictive quality on the holdout set
2. Reviewer-facing interpretability
3. Operational simplicity for downstream app loading

A tree-based model is acceptable if it is the clearest justified winner and the notebook explains its feature drivers and limitations clearly.

## Runtime And Practicality Constraints

- Keep the PyCaret comparison set intentionally small to avoid excessive runtime.
- Avoid notebook code that depends on non-deterministic or unnecessary heavy steps beyond the required comparison.
- Prefer code that can run both in a live notebook and via `nbconvert`.

## Deliverable

When complete, `notebooks/modelling.ipynb` should:

- show concise analysis of the data issues and signal,
- train and compare multiple PyCaret classification models including `lightgbm`,
- justify the chosen baseline in reviewer-oriented terms,
- and save `app/artifacts/model.pkl`.
