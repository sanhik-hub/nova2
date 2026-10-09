from backend.services.reasoning_service import ReasoningService


def test_requested_outputs_present_for_labelled_multi_result():
    result = "{timeOfFlight -> 2.04, maximumHeight -> 5.10}"
    assert ReasoningService._requested_outputs_present(
        ["timeOfFlight", "maximumHeight"], result
    )


def test_requested_outputs_rejects_partial_computation():
    result = "{timeOfFlight -> 2.04}"
    assert not ReasoningService._requested_outputs_present(
        ["timeOfFlight", "maximumHeight"], result
    )


def test_requested_outputs_rejects_empty_or_unlabelled_result():
    assert not ReasoningService._requested_outputs_present(["height"], "")
    assert not ReasoningService._requested_outputs_present(["height"], "5.10")
