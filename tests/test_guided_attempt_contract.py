from backend.services.guided_attempt_service import GuidedAttemptService


def test_one_pass_guided_contract_without_hints():
    raw = (
        "SOLUTION|||Let v=2t|||Set t=3|||v=6
"
        "STEP|||What is v at t=0?|||0|||1|||2|||3
"
        "STEP|||What is v at t=3?|||6|||5|||7|||9
"
    )

    steps = GuidedAttemptService._parse_generated_records(raw, include_hints=False)

    assert len(steps) == 2
    assert all(len(step["options"]) == 4 for step in steps)
    assert all(
        step["options"][step["correct_index"]]["text"] in {"0", "6"}
        for step in steps
    )


def test_one_pass_guided_contract_with_hints():
    raw = (
        "SOLUTION|||Let v=2t|||Set t=3|||v=6
"
        "STEP|||What is v at t=0?|||0|||1|||2|||3|||Use the given relation.
"
        "STEP|||What is v at t=3?|||6|||5|||7|||9|||Substitute t into the relation.
"
    )

    steps = GuidedAttemptService._parse_generated_records(raw, include_hints=True)

    assert len(steps) == 2
    assert steps[0]["hint"] == "Use the given relation."
    assert steps[1]["hint"] == "Substitute t into the relation."
    assert all(
        step["options"][step["correct_index"]]["text"] == expected
        for step, expected in zip(steps, ("0", "6"))
    )


def test_python_owns_correct_index_and_model_does_not_supply_it():
    raw = (
        "STEP|||Find x|||2|||1|||3|||4
"
        "STEP|||Find y|||5|||4|||6|||7
"
    )

    steps = GuidedAttemptService._parse_generated_records(raw, include_hints=False)

    assert [step["correct_index"] for step in steps] == [0, 1]
    assert steps[0]["options"][0]["text"] == "2"
    assert steps[1]["options"][1]["text"] == "5"
