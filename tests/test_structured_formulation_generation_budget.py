import json

from backend.services.reasoning_service import ReasoningService


class CapturingRouter:
    def __init__(self):
        self.kwargs = None

    def supports_tool_calls(self):
        return True

    def generate(self, **kwargs):
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
    assert router.kwargs["tool_choice"] == "required"
    assert router.kwargs["enable_thinking"] is False
    assert "do not write a solution or explain" in router.kwargs["system_prompt"]
    assert result["timings_ms"]["final_llm"] == 0.0
