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
    assert result["timings_ms"]["formulation"] == result["timings_ms"]["planner"]
    assert router.calls == 1


def test_assignment_list_expression_is_normalized_without_repair():
    normalized = ReasoningService._normalize_generated_calculate_expression(
        "calculate",
        {"expression": "timeOfFlight = 2; maximumHeight = 5; {timeOfFlight, maximumHeight}"},
        ["timeOfFlight", "maximumHeight"],
    )
    assert normalized["expression"] == "List[Rule[timeOfFlight, 2], Rule[maximumHeight, 5]]"


def test_assignment_normalizer_does_not_relabel_unrequested_symbols():
    arguments = {"expression": "timeOfFlight = 2; maximumHeight = 5; {timeOfFlight, maximumHeight}"}
    normalized = ReasoningService._normalize_generated_calculate_expression(
        "calculate", arguments, ["time of flight", "range"]
    )
    assert normalized is arguments


def test_detailed_value_only_template_is_rejected_for_fallback():
    assert not ReasoningService._answer_template_meets_mode(
        "Time: {{timeOfFlight}} s. Height: {{maximumHeight}} m.",
        "Find time of flight and maximum height for a projectile at 20 m/s.",
        ["timeOfFlight", "maximumHeight"],
        "detailed",
    )
    assert ReasoningService._answer_template_meets_mode(
        "Using the governing relation, calculate each quantity: T = 2u/g = {{timeOfFlight}} s; H = u^2/(2g) = {{maximumHeight}} m.",
        "Find time of flight and maximum height for a projectile at 20 m/s.",
        ["timeOfFlight", "maximumHeight"],
        "detailed",
    )


def test_concise_compact_equations_pass_without_fixed_equation_count():
    template = (
        "Time of flight: $T = 2v_0\\sin(\\theta)/g = {{timeOfFlight}}\\,\\text{s}$. "
        "Maximum height: $H = (v_0\\sin(\\theta))^2/(2g) = {{maximumHeight}}\\,\\text{m}$."
    )
    assert ReasoningService._answer_template_meets_mode(
        template,
        "Find time of flight and maximum height for a projectile launched at 20 m/s.",
        ["timeOfFlight", "maximumHeight"],
        "concise",
    )


def test_concise_value_only_template_is_rejected_for_numeric_problem():
    assert not ReasoningService._answer_template_meets_mode(
        "Time: {{timeOfFlight}} s. Height: {{maximumHeight}} m.",
        "Find time of flight and maximum height for a projectile launched at 20 m/s.",
        ["timeOfFlight", "maximumHeight"],
        "concise",
    )


def test_template_with_missing_or_extra_output_placeholder_is_rejected():
    problem = "Calculate two requested values from the supplied numerical data: 4 and 8."
    assert not ReasoningService._answer_template_meets_mode(
        "Result: $x = {{firstValue}}$.",
        problem,
        ["firstValue", "secondValue"],
        "concise",
    )
    assert not ReasoningService._answer_template_meets_mode(
        "Results: $x = {{firstValue}}$ and $y = {{secondValue}}$; note {{extraValue}}.",
        problem,
        ["firstValue", "secondValue"],
        "concise",
    )


def test_rejected_template_reports_fallback_metadata():
    class Router:
        def generate(self, **kwargs):
            return {
                "success": True,
                "result": (
                    '{"operation":"calculate",'
                    '"arguments":{"expression":"List[Rule[timeOfFlight, 2.04],'
                    'Rule[maximumHeight, 5.10]]"},'
                    '"requested_outputs":["timeOfFlight","maximumHeight"],'
                    '"answer_template":"Time: {{timeOfFlight}} s. Height: {{maximumHeight}} m."}'
                ),
            }

    service = ReasoningService(router=Router())
    service._validate_tool_call = lambda call: {
        "valid": True,
        "tool": call["tool"],
        "arguments": call["arguments"],
    }
    service._execute_tool = lambda operation, arguments: {
        "success": True,
        "result": "List[Rule[timeOfFlight, 2.04], Rule[maximumHeight, 5.10]]",
    }
    service._final_explanation = lambda **kwargs: {
        "success": True,
        "answer": "fallback answer",
        "_final_llm_ms": 12.0,
    }

    result = service._solve_llm_wolfram(
        "Find time of flight and maximum height for a projectile launched at 20 m/s.",
        explanation="concise",
    )

    assert result["answer_rendering"] == "llm_fallback"
    assert result["llm_passes"] == 2
    assert result["timings_ms"]["final_llm"] == 12.0
    assert result["timings_ms"]["formulation"] == result["timings_ms"]["planner"]


def test_relation_for_one_output_cannot_validate_another_value_only_output():
    template = (
        "Time: $T = 2v_0/g = {{timeOfFlight}}$; "
        "Maximum height: {{maximumHeight}} m."
    )
    assert not ReasoningService._answer_template_meets_mode(
        template,
        "Find time of flight and maximum height for a projectile launched at 20 m/s.",
        ["timeOfFlight", "maximumHeight"],
        "concise",
    )


def test_assignment_sequence_with_trailing_semicolon_is_normalized():
    expression = "time_of_flight = 2; maximum_height = 5;"
    result = ReasoningService._assignment_sequence_to_labelled_expression(expression)
    assert result is not None
    assert "Rule[timeOfFlight, 2]" in result
    assert "Rule[maximumHeight, 5]" in result



def test_failed_multi_output_calculation_stops_after_one_repair():
    class Router:
        calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {
                    "success": True,
                    "result": (
                        '{"operation":"calculate",'
                        '"arguments":{"expression":"bad_expression("},'
                        '"requested_outputs":["timeOfFlight","maximumHeight"],'
                        '"answer_template":"Time: {{timeOfFlight}} s; '
                        'height: {{maximumHeight}} m"}'
                    ),
                }
            return {
                "success": True,
                "result": (
                    '{"operation":"solve","arguments":{"equation":"x==2",'
                    '"variable":"x"},'
                    '"requested_outputs":["timeOfFlight","maximumHeight"]}'
                ),
            }

    router = Router()
    service = ReasoningService(router=router)
    service._validate_tool_call = lambda call: {
        "valid": True,
        "tool": call["tool"],
        "arguments": call["arguments"],
    }
    executions = []

    def execute(operation, arguments):
        executions.append((operation, arguments))
        if len(executions) == 1:
            return {"success": False, "error": "synthetic invalid expression"}
        return {"success": True, "result": "{timeOfFlight -> 2.04}"}

    service._execute_tool = execute
    result = service._solve_llm_wolfram(
        "Find the time of flight and maximum height for a projectile.",
        explanation="concise",
    )

    assert result["success"] is False
    assert "incomplete result" in result["error"].lower()
    assert len(router.calls) == 2  # initial formulation + exactly one repair
    repair_payload = router.calls[1]["prompt"]
    assert "timeOfFlight" in repair_payload
    assert "maximumHeight" in repair_payload
    assert len(executions) == 2
