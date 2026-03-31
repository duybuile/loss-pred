import pytest


def make_prediction(probability_of_loss, flags=None, warnings=None):
    return {
        "is_loss_making_prediction": probability_of_loss >= 0.5,
        "probability_of_loss": probability_of_loss,
        "top_features": ["premium_to_limit", "prior_claims"],
        "input_quality_flags": flags or [],
        "model_warnings": warnings or [],
    }


def make_config():
    return {
        "high_confidence_threshold": 0.30,
        "medium_confidence_threshold": 0.15,
        "retrieval_n_results": 3,
        "retrieval_distance_threshold": 0.5,
    }


def make_controller(record_outcomes=None):
    from app.controller import ControllerPolicy

    return ControllerPolicy(config=make_config(), record_outcomes=record_outcomes or {})


class TestConfidenceLevel:
    def test_high_confidence_above_threshold(self):
        ctrl = make_controller()
        assert ctrl.compute_confidence(make_prediction(0.85)) == "high"

    def test_high_confidence_below_zero_point_five(self):
        ctrl = make_controller()
        assert ctrl.compute_confidence(make_prediction(0.15)) == "high"

    def test_medium_confidence(self):
        ctrl = make_controller()
        assert ctrl.compute_confidence(make_prediction(0.68)) == "medium"

    def test_low_confidence_near_boundary(self):
        ctrl = make_controller()
        assert ctrl.compute_confidence(make_prediction(0.52)) == "low"

    def test_quality_flag_downgrades_medium_to_low(self):
        ctrl = make_controller()
        pred = make_prediction(0.68, flags=["optional_field_missing:broker"])
        assert ctrl.compute_confidence(pred) == "low"

    def test_required_field_missing_forces_low(self):
        ctrl = make_controller()
        pred = make_prediction(0.85, flags=["required_field_missing:limit"])
        assert ctrl.compute_confidence(pred) == "low"

    def test_model_warning_forces_low(self):
        ctrl = make_controller()
        pred = make_prediction(0.85, warnings=["feature_missing_after_transform:some_col"])
        assert ctrl.compute_confidence(pred) == "low"


class TestRetrievalGating:
    def test_no_retrieval_for_high_confidence(self):
        ctrl = make_controller()
        assert ctrl.should_retrieve(confidence_level="high", predict_failed=False) is False

    def test_no_retrieval_for_medium_confidence(self):
        ctrl = make_controller()
        assert ctrl.should_retrieve(confidence_level="medium", predict_failed=False) is False

    def test_retrieval_for_low_confidence(self):
        ctrl = make_controller()
        assert ctrl.should_retrieve(confidence_level="low", predict_failed=False) is True

    def test_retrieval_when_predict_failed(self):
        ctrl = make_controller()
        assert ctrl.should_retrieve(confidence_level="high", predict_failed=True) is True


class TestConflictDetection:
    def test_no_conflict_when_outcomes_agree(self):
        outcomes = {"REC_001": True, "REC_002": True, "REC_003": True}
        ctrl = make_controller(record_outcomes=outcomes)
        retrieval = [
            {"record_id": "REC_001", "document": "...", "distance": 0.1},
            {"record_id": "REC_002", "document": "...", "distance": 0.2},
            {"record_id": "REC_003", "document": "...", "distance": 0.3},
        ]
        assert ctrl.detect_conflict(is_loss_making_prediction=True, retrieval_results=retrieval) is False

    def test_conflict_when_outcomes_disagree(self):
        outcomes = {"REC_001": False, "REC_002": False, "REC_003": True}
        ctrl = make_controller(record_outcomes=outcomes)
        retrieval = [
            {"record_id": "REC_001", "document": "...", "distance": 0.1},
            {"record_id": "REC_002", "document": "...", "distance": 0.2},
            {"record_id": "REC_003", "document": "...", "distance": 0.3},
        ]
        assert ctrl.detect_conflict(is_loss_making_prediction=True, retrieval_results=retrieval) is True

    def test_no_conflict_on_poor_retrieval_quality(self):
        """When all retrieved records have high distance, conflict detection is skipped."""
        outcomes = {"REC_001": False, "REC_002": False}
        ctrl = make_controller(record_outcomes=outcomes)
        retrieval = [
            {"record_id": "REC_001", "document": "...", "distance": 0.8},
            {"record_id": "REC_002", "document": "...", "distance": 0.9},
        ]
        assert ctrl.detect_conflict(is_loss_making_prediction=True, retrieval_results=retrieval) is False


class TestRiskAssessment:
    def test_likely_loss(self):
        ctrl = make_controller()
        assert ctrl.compute_risk_assessment(is_loss_making=True, confidence_level="high") == "likely_loss"

    def test_unlikely_loss(self):
        ctrl = make_controller()
        assert ctrl.compute_risk_assessment(is_loss_making=False, confidence_level="medium") == "unlikely_loss"

    def test_unclear_when_low_confidence(self):
        ctrl = make_controller()
        assert ctrl.compute_risk_assessment(is_loss_making=True, confidence_level="low") == "unclear"


class TestEscalation:
    def test_escalation_on_low_confidence_no_retrieval(self):
        ctrl = make_controller()
        assert ctrl.should_escalate(
            confidence_level="low",
            conflict_detected=False,
            retrieval_ran=False,
        ) is True

    def test_no_escalation_on_high_confidence(self):
        ctrl = make_controller()
        assert ctrl.should_escalate(
            confidence_level="high",
            conflict_detected=False,
            retrieval_ran=False,
        ) is False

    def test_escalation_on_conflict(self):
        ctrl = make_controller()
        assert ctrl.should_escalate(
            confidence_level="medium",
            conflict_detected=True,
            retrieval_ran=True,
        ) is True

    def test_escalation_on_low_confidence_after_retrieval(self):
        ctrl = make_controller()
        assert ctrl.should_escalate(
            confidence_level="low",
            conflict_detected=False,
            retrieval_ran=True,
        ) is True
