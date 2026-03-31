# Approach

## Key decisions

**Model: Logistic Regression over tree-based alternatives.**
PyCaret compared ~15 classifiers. LR won on a combination of AUC, calibration, and fit-for-purpose properties: the dataset is small (~hundreds of records), which makes gradient-boosted trees prone to overfitting; LR's coefficients give the reviewer a directional, stable signal rather than a black-box importance score. The top drivers — `claims_per_year`, `risk_type_cyber/property`, `territory_US` — align with underwriting intuition, which matters for trust. Notably, `years_trading_missing` ranked second by coefficient magnitude: records where the counterparty's trading history is unknown are measurably riskier than those where it is known. That signal would be lost if missing data were simply imputed to zero.

**Hybrid controller, not a pure LLM agent.**
A tool-calling LLM loop gives the model control over which tools to call and when to stop. That makes cost and latency unpredictable, and makes routing decisions untestable. The controller instead owns all of these decisions deterministically: always run the model first, retrieve only on low confidence, escalate on conflict or missing data. The LLM's role is narrowed to prose synthesis — the one thing it is genuinely better at than rules.

**Retrieval as a cost gate.**
Embedding retrieval adds ~1–2 seconds and a non-trivial inference cost. Running it on every request would be wasteful for the ~60% of records where the model is high-confidence. Gating on confidence level means retrieval fires precisely when historical context is most useful — when the model is uncertain.

**No raw probability in the response.**
A reviewer seeing `probability: 0.61` has no action to take. `medium / likely_loss` tells them both direction and how much to trust it. The calibration gate in the eval harness enforces the same principle: a confident answer on a borderline case fails, even if the direction is correct.

---

## What I would do differently with more time

Threshold optimisation. The current 0.5 decision boundary assumes equal cost for false positives and false negatives. In practice, a missed loss (false negative) is more expensive than a wrongly flagged record. I would tune the operating threshold against a cost matrix derived from actual claim outcomes. I would also add SHAP values per prediction rather than static feature rankings from model coefficients.

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
