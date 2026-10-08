from backend.services.complexity_classifier import ComplexityClassifier


def test_simple_request_keeps_thinking_off():
    assessment = ComplexityClassifier().analyze("What is the molecular geometry of NH3?")
    assert assessment.thinking is False
    assert assessment.level in {"simple", "moderate"}


def test_structurally_complex_request_enables_thinking():
    problem = (
        "A body is launched from height 20 m with speed 30 m/s at 45 degrees. "
        "Determine the time of flight, maximum height, horizontal range, and "
        "velocity immediately before impact, then verify the result using the "
        "governing equations."
    )
    assessment = ComplexityClassifier().analyze(problem)
    assert assessment.thinking is True
    assert assessment.level == "high"
    assert assessment.score >= ComplexityClassifier.THINKING_THRESHOLD


def test_classifier_is_domain_agnostic():
    classifier = ComplexityClassifier()
    math_case = classifier.analyze(
        "Find x and y from 2x + 3y = 7 and 4x - y = 5, then verify both equations."
    )
    chemistry_case = classifier.analyze(
        "For a reaction with 2.0 mol A and 3.0 mol B, determine limiting reagent, "
        "product amount, and remaining excess, then verify atom balance."
    )
    assert math_case.thinking is True
    assert chemistry_case.thinking is True


def test_complexity_does_not_depend_on_subject_names():
    classifier = ComplexityClassifier()
    generic = classifier.analyze(
        "Given 2 quantities and 3 constraints, determine two unknowns and verify "
        "the result by substituting back into every constraint."
    )
    assert generic.thinking is True
