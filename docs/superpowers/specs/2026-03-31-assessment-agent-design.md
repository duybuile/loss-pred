# Assessment Agent Design

## Goal

Design the remaining app work for the take-home so the assessment system behaves like a production-minded reviewer support tool rather than a demo agent. The agent should produce a useful recommendation for a reviewer, use model scoring as its primary evidence source, use retrieval only when uncertainty justifies the extra cost, and escalate to a human when evidence is weak or contradictory.

## Scope

This spec covers the remaining app-facing design for:

- the assessment agent control flow in `app/agent.py`
- the model tool contract in `app/model.py` and `app/tools.py`
- the reviewer-facing response contract in `app/schemas.py`

This spec does not cover:

- the modelling notebook implementation in `notebooks/modelling.ipynb`
- the training workflow that creates `app/artifacts/model.pkl`
- the evaluation harness design in `evals/`

## Product Position

The system is a decision-support tool for non-technical reviewers, not an autonomous approval engine. Its job is to surface a directional recommendation, summarize the main evidence, and state clearly when the case needs a second opinion.

The design should optimize for:

- predictable behavior under uncertainty
- bounded cost and latency
- concise reviewer-facing output
- explicit escalation when the system should not be trusted to generalize

The design should not optimize for:

- maximum tool usage
- conversational sophistication for its own sake
- presenting precise-looking numbers when the system cannot justify them

## Recommended Architecture

The agent should use a hybrid controller architecture. The LLM should not be the primary decision-maker for tool routing or conflict resolution. Instead, deterministic logic should decide when to call tools, when to stop, and when to escalate. The LLM should only be used when it adds value by turning coherent evidence into concise reviewer-facing language.

### Responsibilities

#### 1. Prediction step

The controller should call `predict_loss` first. This tool is the primary evidence source and should return structured predictive evidence derived from the saved model artifact.

#### 2. Policy step

The controller in `app/agent.py` should evaluate the model output and determine:

- the operational `confidence_level`
- whether retrieval is warranted
- whether the case should escalate to a human reviewer
- whether the final answer should be LLM-written or assembled deterministically

#### 3. Retrieval step

The controller should call `retrieve_similar_records` only when the model signal is too weak or unavailable to support a reliable recommendation on its own.

#### 4. Response step

If the evidence is coherent, the LLM may synthesize the structured evidence into a short recommendation. If the evidence is weak, degraded, or contradictory, the final response should be built deterministically and should recommend human review rather than asking the LLM to reconcile uncertainty.

## Why This Architecture

This approach fits the take-home and still reads as production-minded.

- It preserves the agent framing expected by the README.
- It prevents the LLM from deciding policy questions that should be deterministic.
- It keeps cost and latency bounded by making retrieval conditional.
- It provides a clean place to encode escalation behavior that can be tested directly.

It is stronger than a prompt-only design because key safety and budget decisions do not depend on whether the model follows instructions perfectly.

## Tool Design

### `predict_loss`

`predict_loss` should remain a narrow tool that returns model-derived facts. It should not compute the final `confidence_level`, because confidence in this design is an operational trust signal rather than a pure model output.

Recommended `predict_loss` output:

- `is_loss_making_prediction`
- `probability_of_loss`
- `top_features`
- `input_quality_flags`
- `model_warnings`

Field intent:

- `probability_of_loss` is the raw probability that the record is loss-making.
- `top_features` provides concise drivers that the controller or LLM can turn into reviewer-facing reasons.
- `input_quality_flags` captures missing, malformed, suspicious, or unsupported input conditions.
- `model_warnings` captures artifact, preprocessing, or runtime issues that reduce trust.

`predict_loss` should fail gracefully. If the model artifact is missing or inference fails, the tool should return a structured error or warning payload rather than crashing the whole assessment loop.

### `retrieve_similar_records`

Retrieval should be fallback-only. It should not run by default for every record. Its role is to provide supporting historical analogs when model evidence is too weak, missing, or potentially misleading.

Retrieval usage rules:

- default to a small `n_results`
- cap the maximum number of retrieved records
- allow at most one retrieval round per request
- treat weak retrieval as uncertainty, not as a reason to keep searching

The agent should not dump raw retrieved documents into the main response. It should summarize whether comparable historical records support, weaken, or complicate the recommendation.

## Controller Policy

The controller should own the core decision logic.

### Model-first flow

The default flow should be:

1. Call `predict_loss`
2. Compute `confidence_level`
3. Decide whether retrieval is necessary
4. Decide whether to escalate
5. Produce a final response

This keeps the cheapest and most direct evidence source first in the path.

### Retrieval gating

Retrieval should be triggered only when:

- the model confidence is low
- the model tool fails or returns a serious warning
- the record has enough quality issues that the model output should not stand alone

Retrieval should not run for high-confidence model outputs just to make the answer feel richer. That would increase cost, latency, and prompt surface area without reliably improving decision quality.

### Conflict policy

If model evidence and retrieval evidence point in different directions, the system should escalate to human review. The LLM should not be allowed to smooth over the disagreement with persuasive prose.

## Probability Of Loss Versus Confidence Level

The design should treat `probability_of_loss` and `confidence_level` as different signals.

### `probability_of_loss`

This is the model's estimate of how likely the record is to be loss-making.

### `confidence_level`

This is the controller's assessment of how much the system should trust the model output enough to act on it without additional evidence.

Confidence should therefore be computed in the controller layer, not inside `predict_loss`.

## Confidence Policy

The controller should compute confidence from deterministic rules.

### Base confidence from probability margin

Let:

- `pl = probability_of_loss`
- `margin = abs(pl - 0.5)`

Recommended base bands:

- `high` if `margin >= 0.30`
- `medium` if `0.15 <= margin < 0.30`
- `low` if `margin < 0.15`

This avoids conflating risk with certainty. A prediction close to `0.5` is less actionable than one far from the threshold, regardless of which side of the threshold it lands on.

### Confidence adjustments

After computing the base confidence, the controller should adjust it downward when trust conditions are weaker.

Recommended adjustments:

- downgrade one level when non-critical `input_quality_flags` are present
- force `low` when required fields are missing
- force `low` when values are invalid or clearly suspicious
- force `low` when `model_warnings` indicate degraded reliability

Retrieval should not increase confidence by itself. It may provide useful context, but it should not turn an inherently uncertain model output into a high-confidence conclusion.

If retrieval conflicts with the model, confidence should remain at most `low` and the system should escalate.

## Response Contract

The response schema should be structured for actionability and auditability rather than raw detail.

Recommended top-level fields:

- `record_id`
- `recommendation`
- `risk_assessment`
- `confidence_level`
- `summary`
- `key_factors`
- `similar_records_summary`
- `second_opinion_recommended`
- `review_guidance`
- `tools_used`
- `warnings`

### Field intent

- `recommendation`: the main reviewer-facing recommendation in plain English
- `risk_assessment`: normalized directional output such as `likely_loss`, `unlikely_loss`, or `unclear`
- `confidence_level`: `high`, `medium`, or `low`
- `summary`: short supporting explanation
- `key_factors`: concise drivers from the model and, when used, retrieval
- `similar_records_summary`: optional text populated only when retrieval runs
- `second_opinion_recommended`: explicit escalation flag
- `review_guidance`: the next action for the reviewer
- `tools_used`: auditability and debugging support
- `warnings`: known limitations in the assessment path

Raw probability should not be a primary reviewer-facing field. It may be useful internally, but the reviewer-facing contract should favor calibrated bands and actionable language over false precision.

## Error Handling And Guardrails

The system should fail toward explicit uncertainty rather than overconfident output.

Recommended handling:

- if the model artifact cannot be loaded, mark model evidence unavailable and try retrieval once
- if retrieval produces weak or irrelevant analogs, return `risk_assessment = unclear`
- if model and retrieval conflict, set `second_opinion_recommended = true`
- if required fields are missing or invalid, lower confidence and add warnings
- if the LLM is used, limit it to a single synthesis pass from structured evidence
- do not allow repeated tool loops while trying to manufacture certainty

These rules should be encoded in the controller, not only in prompt wording.

## LLM Role

The LLM should have a narrow role:

- convert coherent structured evidence into concise reviewer-facing prose
- keep the recommendation readable and non-technical
- summarize evidence without exposing hidden chain-of-thought

The LLM should not:

- decide whether to retrieve
- decide whether conflicting evidence is acceptable
- repeatedly call tools seeking certainty
- replace explicit escalation with smooth but weak language

## Testing Strategy

The main testing target should be controller behavior, because that is where the product judgment lives.

Recommended scenarios:

- high-confidence model prediction does not trigger retrieval
- near-boundary model output triggers retrieval
- missing required fields downgrade confidence and add warnings
- model failure falls back to retrieval
- weak retrieval yields `unclear` plus escalation
- model and retrieval disagreement forces `second_opinion_recommended = true`
- response schema remains stable and audit-friendly across clean and degraded cases

This test focus shows that the system has defined behavior under uncertainty instead of only a happy path.

## Implementation Guidance

When implementing this design:

- keep `app/model.py` responsible for model loading and raw prediction outputs
- keep `app/tools.py` responsible for tool contracts and dispatch
- put confidence computation, retrieval gating, and escalation policy in `app/agent.py`
- update `app/schemas.py` to reflect the reviewer-facing contract

This keeps the boundaries clear and makes future policy changes cheaper than changing the model interface.

## Success Criteria

The design is successful if the final implementation shows:

- model-first, bounded tool usage
- fallback-only retrieval
- deterministic escalation on conflict or degraded evidence
- reviewer-facing output that is concise and actionable
- a clean separation between model facts and controller policy
- testable behavior under uncertainty

## Explicit Non-Goals

This design intentionally avoids:

- a fully LLM-driven multi-step reasoning loop
- retrieval on every request
- using retrieval to override uncertainty with persuasive narrative
- exposing raw evidence dumps as the main reviewer interface
- building a more complex orchestration framework than the take-home needs
