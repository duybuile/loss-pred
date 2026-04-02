# Approach

## Key decisions

**Model: Logistic Regression over tree-based alternatives.**
Initially, we wanted to use PyCaret to compare ~15 classifiers. However, there was some dependency issues, we decided to build 3 different models from scratch: Logistic Regression, Random Forest and LightGBM. LR won on a combination of AUC, calibration, and fit-for-purpose properties: the dataset is small (~hundreds of records), which makes gradient-boosted trees prone to overfitting; LR's coefficients give the reviewer a directional, stable signal rather than a black-box importance score. The top drivers — `claims_per_year`, `risk_type_cyber/property`, `territory_US` — align with underwriting intuition, which matters for trust. Notably, `years_trading_missing` ranked second by coefficient magnitude: records where the counterparty's trading history is unknown are measurably riskier than those where it is known. That signal would be lost if missing data were simply imputed to zero.

**Hybrid controller, not a pure LLM agent.**
A tool-calling LLM loop gives the model control over which tools to call and when to stop. That makes cost and latency unpredictable, and makes routing decisions untestable. The controller instead owns all of these decisions deterministically: always run the model first, retrieve only on low confidence, escalate on conflict or missing data. The LLM's role is narrowed to prose synthesis — the one thing it is genuinely better at than rules.

**Retrieval as a cost gate.**
Embedding retrieval adds ~1–2 seconds and a non-trivial inference cost. Running it on every request would be wasteful for the ~60% of records where the model is high-confidence. Gating on confidence level means retrieval fires precisely when historical context is most useful — when the model is uncertain.

**No raw probability in the response.**
A reviewer seeing `probability: 0.61` has no action to take. `medium / likely_loss` tells them both direction and how much to trust it. The calibration gate in the eval harness enforces the same principle: a confident answer on a borderline case fails, even if the direction is correct.

---

## What I would do differently with more time

### Model improvements
- **Feature selection:** With more time, I would experiment with adding back some of the features that were dropped for simplicity (e.g. `industry_code`, `broker_id`) using regularization to prevent overfitting. I would also consider dimensionality reduction techniques (e.g. PCA) if adding many one-hot encoded variables.
- **Feature engineering:** Create interaction terms (e.g. `premium_to_limit * risk_type_cyber`) and polynomial features (e.g. `claims_per_year^2`) to capture non-linear relationships. Experiment with binning continuous variables to see if that improves model stability and interpretability.
- **Ensemble approach:** Combine the logistic regression with a tree-based model (e.g. a simple average of predicted probabilities) to see if it improves AUC without sacrificing interpretability too much. The tree-based model could capture non-linearities and interactions that LR misses, while LR provides a stable, interpretable backbone.
- **Hyperparameter tuning:** Perform a more exhaustive search over regularization strength and class weights to optimize for the specific cost structure of false positives vs false negatives in this use case.
- **LIME or SHAP explanations:** Add local explanation methods to provide case-specific insights into why the model made a particular prediction, which could be more actionable for reviewers than global feature importance.

### Controller improvements
- **Dynamic confidence thresholds:** Instead of fixed thresholds for `high`, `medium`, and `low`, implement a dynamic thresholding mechanism that adjusts based on recent model performance and feedback from reviewers. For example, if the model has been consistently accurate on `medium` confidence cases, the threshold for `high` confidence could be raised to reduce unnecessary retrievals.
- **Multi-stage retrieval:** For `medium` confidence cases, implement a tiered retrieval system where a cheaper, more focused retrieval is attempted first (e.g. retrieving only the most similar 3 records), and if that doesn't yield a confident second opinion, escalate to a more comprehensive retrieval.
- **Cost-aware routing:** Incorporate a cost model that estimates the expected cost of retrieval and synthesis for each request, and use that to make more informed routing decisions. For example, if the estimated cost of retrieval is high and the confidence is borderline, the controller could choose to provide a more conservative recommendation rather than escalating to retrieval.
- **LangGraph integration:** Use a LangGraph to track the flow of information and decisions across the different components (model, retrieval, LLM) for each case. This would enable better debugging, monitoring, and understanding of where errors or bottlenecks occur in the process.

### LLM improvements
- **Prompt engineering:** 
    - Refine the prompts used for synthesis and the judge to better align with the specific language and decision criteria of insurance underwriting. For example, the synthesis prompt could be tailored to explicitly ask for actionable insights rather than just a summary of the retrieved records.
    - Prompt registry: Maintain a registry of different prompt versions and their performance metrics, allowing for systematic A/B testing and iteration on prompt design.
- **Few-shot examples:** Provide the LLM with a few examples of past cases, including the model's prediction, the retrieved records, and the final reviewer decision, to help it learn the patterns of when to trust the model vs when to defer to the retrieved context.
- **LLM evaluation:** Implement a more systematic evaluation of the LLM's performance in synthesis and judging, using a separate validation set of cases with known outcomes to ensure that it is providing accurate and helpful insights rather than just plausible-sounding text.

### Frontend improvements
- **Interactive explanations:** In the UI, provide interactive explanations of the model's prediction and the retrieved records, allowing reviewers to drill down into the specific features and historical cases that influenced the recommendation.
- **Feedback capture:** Add a mechanism for reviewers to provide feedback on the recommendation directly in the UI, which would then be fed back into the model training and evaluation process to create a continuous improvement

---

## Where the AI tool made a suggestion I overrode

The agent was initially scaffolded as a pure LLM tool-calling loop. I replaced this with the hybrid controller because the loop gave the LLM unnecessary authority over routing and cost decisions that are better owned in code. The AI also generated `model_warnings` as a Python list embedded in a single string (e.g. `"feature_missing:['col1', 'col2']"`); I changed this to one flag per column (`feature_missing_after_transform:col1`) so the controller could parse and act on them without string manipulation. These were not stylistic preferences — they affected correctness and testability.

---

## How I would know if the model is degrading

**Distribution shift signals:**
- Prediction distribution: if the share of `likely_loss` predictions drifts significantly from the historical base rate, the model has likely encountered a covariate shift.
- Confidence distribution: an increasing proportion of `low` confidence outputs indicates the model is operating outside its training envelope.
- Feature distributions: `premium_to_limit` and `claims_per_year` are the model's strongest predictors. If their distributions shift — new business lines, rate changes — the decision boundary learned from historical data is no longer valid.
- Outcome calibration: once outcomes are known (loss ratios close out), compare predicted confidence to actual accuracy. A calibrated model should be right ~90% of the time on `high` confidence predictions. If that drops, the model needs retraining.

---

## Feedback loop design

Capture three things alongside each assessment: the recommendation, the reviewer's final decision, and whether the reviewer overrode the system. Overrides are the primary signal — they reveal where the system is systematically wrong, not just occasionally unlucky.

Periodic retraining should use reviewer decisions as labels, but weighted: reviewer quick-approvals of high-confidence recommendations carry less label value than deliberate overrides of low-confidence ones. Track disagreement patterns by risk type, territory, and broker to catch systematic blind spots (e.g., "the model underestimates cyber risk in LATAM").

The feedback loop should also inform the eval harness: if reviewers consistently override recommendations on borderline cases that the judge marks as `calibration_ok`, the judge prompt needs updating.

---

## Pre-production requirements

**Cost:** Set a per-assessment budget ceiling (LLM synthesis + judge). Alert on cost spikes — an unexpected surge in retrieval calls usually means confidence is degrading across the portfolio.

**Latency:** P95 < 3 s for non-retrieval path, < 8 s for retrieval path. The sentence-transformer embedding model adds ~300 ms on first call (warm) — bake it into the Docker image to avoid cold-start delays.

**Monitoring:** Track `confidence_level` distribution, `second_opinion_rate`, `tools_used` breakdown, and LLM API error rate in a rolling 24-hour window. Dashboard these, not just alert on them.

**Kill switch:** The system should stop making recommendations and defer entirely to human reviewers if: model or pipeline artifacts fail to load; LLM API key is invalid or rate-limited for > N consecutive requests; or `second_opinion_rate` exceeds a configured threshold (e.g. > 40% in a rolling hour — a signal the model is no longer reliable for the incoming portfolio).
