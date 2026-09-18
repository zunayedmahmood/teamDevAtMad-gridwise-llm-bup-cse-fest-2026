import json
from collections import Counter
from pathlib import Path


def test_paraphrase_corpus_has_36_balanced_notes():
    payload = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "paraphrase_cases.json").read_text(
            encoding="utf-8"
        )
    )
    cases = payload["cases"]
    note_count = sum(len(case["notes"]) for case in cases)
    expected_count = sum(len(case["expected"]) for case in cases)

    assert note_count == 36
    assert expected_count == 36

    counts = Counter(
        expected["directive_type"]
        for case in cases
        for expected in case["expected"]
    )
    assert counts == {
        "solar_reduction": 6,
        "minimum_battery_reserve": 6,
        "no_charge_window": 6,
        "no_discharge_window": 6,
        "max_grid_window": 6,
        "no_op": 6,
    }

    for case in cases:
        assert len(case["notes"]) == len(case["expected"])
        assert [item["note_index"] for item in case["expected"]] == list(
            range(len(case["notes"]))
        )
