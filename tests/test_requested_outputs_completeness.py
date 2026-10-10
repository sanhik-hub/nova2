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


def test_assignment_sequence_is_converted_to_labelled_rules():
    expression = (
        "timeOfFlight = (2*20*Sin[30 Degree])/9.8; "
        "maxHeight = (20^2*Sin[30 Degree]^2)/(2*9.8)"
    )
    converted = ReasoningService._assignment_sequence_to_labelled_expression(
        expression
    )
    assert converted is not None
    assert converted.startswith("List[Rule[timeOfFlight,")
    assert "Rule[maxHeight," in converted
    assert "==" not in converted
    assert ";" not in converted


def test_assignment_conversion_does_not_rewrite_a_single_equation():
    assert (
        ReasoningService._assignment_sequence_to_labelled_expression("x = 2")
        is None
    )
    assert (
        ReasoningService._assignment_sequence_to_labelled_expression("x == 2")
        is None
    )
