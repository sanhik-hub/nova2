from backend.services.reasoning_service import ReasoningService


def test_renderer_inserts_only_values_from_labelled_wolfram_result():
    template = (
        "Time of flight: $T = {{timeOfFlight}}\\,\\text{s}$. "
        "Maximum height: $H = {{maximumHeight}}\\,\\text{m}$."
    )
    result = "{timeOfFlight -> 2.0408163265, maximumHeight -> 5.1020408163}"

    rendered = ReasoningService._render_verified_answer_template(
        template,
        ["timeOfFlight", "maximumHeight"],
        result,
    )

    assert rendered == (
        "Time of flight: $T = 2.0408163265\\,\\text{s}$. "
        "Maximum height: $H = 5.1020408163\\,\\text{m}$."
    )


def test_renderer_supports_wolfram_rule_constructor_with_nested_values():
    result = (
        "List[Rule[timeOfFlight, N[(2*20*Sin[30 Degree])/9.8]], "
        "Rule[maximumHeight, N[(20*Sin[30 Degree])^2/(2*9.8)]]]"
    )
    values = ReasoningService._extract_labelled_wolfram_values(result)

    assert values["timeofflight"] == "N[(2*20*Sin[30 Degree])/9.8]"
    assert values["maximumheight"] == "N[(20*Sin[30 Degree])^2/(2*9.8)]"


def test_renderer_rejects_missing_requested_output_placeholder():
    rendered = ReasoningService._render_verified_answer_template(
        "Time: {{timeOfFlight}} s.",
        ["timeOfFlight", "maximumHeight"],
        "{timeOfFlight -> 2.04, maximumHeight -> 5.10}",
    )
    assert rendered is None


def test_renderer_rejects_output_not_returned_by_wolfram():
    rendered = ReasoningService._render_verified_answer_template(
        "Time: {{timeOfFlight}} s. Height: {{maximumHeight}} m.",
        ["timeOfFlight", "maximumHeight"],
        "{timeOfFlight -> 2.04}",
    )
    assert rendered is None


def test_one_pass_computation_uses_template_without_final_generation():
    class Router:
        calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            return {
                "success": True,
                "result": (
                    '{"operation":"calculate",'
                    '"arguments":{"expression":"List[Rule[timeOfFlight, 2.04],'
                    'Rule[maximumHeight, 5.10]]"},'
                    '"requested_outputs":["timeOfFlight","maximumHeight"],'
                    '"answer_template":"Time: {{timeOfFlight}} s. '
                    'Height: {{maximumHeight}} m."}'
                ),
            }

    router = Router()
    service = ReasoningService(router=router)
    service._validate_tool_call = lambda call: {
        "valid": True,
        "tool": call["tool"],
        "arguments": call["arguments"],
    }
    service._execute_tool = lambda operation, arguments: {
        "success": True,
        "result": "List[Rule[timeOfFlight, 2.04], Rule[maximumHeight, 5.10]]",
    }
    service._final_explanation = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("final LLM generation should not run on the one-pass path")
    )

    result = service._solve_llm_wolfram(
        "Find the time of flight and maximum height.",
        explanation="concise",
    )

    assert result["success"] is True
    assert result["answer"] == "Time: 2.04 s. Height: 5.10 m."
    assert result["llm_passes"] == 1
    assert result["answer_rendering"] == "deterministic_template"
    assert result["timings_ms"]["final_llm"] == 0.0
    assert router.calls == 1
