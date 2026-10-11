import json

from backend.services.reasoning_service import ReasoningService


class CapturingRouter:
    def __init__(self):
        self.kwargs = None
        self.calls = 0

    def supports_tool_calls(self):
        return True

    def generate(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return {
            "success": True,
            "result": json.dumps(
                {
                    "operation": "calculate",
                    "arguments": {
                        "expression": "{timeOfFlight -> 2.04, maximumHeight -> 5.10}"
                    },
                    "requested_outputs": ["timeOfFlight", "maximumHeight"],
                }
            ),
        }


def test_complex_result_only_formulation_uses_scoped_structured_budget():
    router = CapturingRouter()
    service = ReasoningService(router=router)
    service._validate_tool_call = lambda call: {
        "valid": True,
        "tool": call["tool"],
        "arguments": call["arguments"],
    }
    service._execute_tool = lambda operation, arguments: {
        "success": True,
        "result": "{timeOfFlight -> 2.04, maximumHeight -> 5.10}",
    }

    result = service._solve_llm_wolfram(
        "Find time of flight and maximum height for a projectile.",
        explanation="off",
    )

    assert result["success"] is True
    assert router.kwargs["max_tokens"] == ReasoningService.STRUCTURED_TOOL_TOKENS
    assert router.kwargs["max_tokens"] == 256
    assert router.kwargs["response_format"]["type"] == "json_object"
    schema = router.kwargs["response_format"]["schema"]
    assert schema["required"] == ["operation", "arguments", "requested_outputs"]
    expression_description = schema["properties"]["arguments"]["properties"]["expression"]["description"]
    assert "Executable Wolfram Language syntax only" in expression_description
    assert "not explanation or derivation prose" in expression_description
    assert "no With/local assignments" in expression_description
    assert "tool_choice" not in router.kwargs
    assert router.kwargs["enable_thinking"] is False
    assert "do not write a solution or explain" in router.kwargs["system_prompt"].lower()
    assert "computational stem request" in router.kwargs["system_prompt"].lower()
    assert "projectile" not in router.kwargs["system_prompt"].lower()
    assert result["timings_ms"]["final_llm"] == 0.0
    assert router.calls == 1


def test_result_only_mode_does_not_regenerate_for_missing_requested_output():
    router = CapturingRouter()
    service = ReasoningService(router=router)
    service._validate_tool_call = lambda call: {
        "valid": True,
        "tool": call["tool"],
        "arguments": call["arguments"],
    }
    service._execute_tool = lambda operation, arguments: {
        "success": True,
        "result": "{timeOfFlight -> 2.04}",
    }

    result = service._solve_llm_wolfram(
        "Find time of flight and maximum height for a projectile.",
        explanation="off",
    )

    assert result["success"] is False
    assert "no second model generation was attempted" in result["error"]
    assert router.calls == 1


class RepairingRouter:
    def __init__(self):
        self.calls = []

    def supports_tool_calls(self):
        return False

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        expression = (
            "With[{v = 20, Î¸ = 30 Degree, g = 9.8}, "
            "{timeOfFlight -> (2*v*Sin[Î¸])/g, "
            "maximumHeight -> (v^2*Sin[Î¸]^2)/g}]}}}</think>"
            if len(self.calls) == 1
            else "{timeOfFlight -> (2*20*Sin[30 Degree])/9.8, "
                 "maximumHeight -> (20^2*Sin[30 Degree]^2)/(2*9.8)}"
        )
        return {
            "success": True,
            "result": json.dumps({
                "operation": "calculate",
                "arguments": {"expression": expression},
                "requested_outputs": ["timeOfFlight", "maximumHeight"],
            }),
        }


def test_result_only_mode_does_not_regenerate_after_invalid_expression():
    router = RepairingRouter()
    service = ReasoningService(router=router)
    executed = []

    def execute(operation, arguments):
        executed.append(arguments["expression"])
        return {
            "success": True,
            "result": "{timeOfFlight -> 2.0408, maximumHeight -> 5.1020}",
        }

    service._execute_tool = execute

    result = service._solve_llm_wolfram(
        "A projectile is launched at 20 m/s at 30 degrees. Take g = 9.8 m/s^2. "
        "Find time of flight and maximum height.",
        explanation="off",
    )

    assert result["success"] is False
    assert "no second model generation was attempted" in result["error"]
    assert len(router.calls) == 1
    assert executed == []

def test_assignment_sequence_with_final_output_list_is_normalized_without_side_effects():
    service = ReasoningService(router=CapturingRouter())
    expression = (
        "v0 = 20; angle = 30 Degree; g = 9.8; "
        "time_of_flight = (2 * v0 * Sin[angle]) / g; "
        "maximum_height = (v0 * Sin[angle])^2 / (2 * g); "
        "{time_of_flight, maximum_height}"
    )

    normalized = service._assignment_sequence_to_labelled_expression(expression)

    assert normalized is not None
    assert normalized.startswith("List[Rule[timeOfFlight,")
    assert "Rule[maximumHeight," in normalized
    assert "time_of_flight" not in normalized
    assert "maximum_height" not in normalized
    assert "v0" not in normalized
    assert "angle" not in normalized
    assert "g" not in normalized
    assert ";" not in normalized

