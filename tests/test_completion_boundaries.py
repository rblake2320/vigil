import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from watcher import Procedure, Watcher
from perception.coach import PerceptionCoach
from perception.mobile_watch import load_procedure


@pytest.mark.parametrize("description", [
    "MATCH: NO. The saved successfully message is not visible.",
    "MATCH: YES saved successfully", "", '{"success":true}',
])
def test_prose_never_advances(description):
    p = Procedure.__new__(Procedure)
    p.steps = [{"id": 1, "description": "Save", "detect": "saved successfully"}]
    p.current_step_idx = 0
    w = Watcher.__new__(Watcher)
    w.procedure = p
    w._speak = lambda _: None
    w._handle_coach(description)
    assert not p.is_complete


def test_window_title_does_not_prove_saved_file():
    coach = PerceptionCoach.__new__(PerceptionCoach)
    assert coach._check_step_from_signals({"signals": ["saved"]},
                                        {"window_title": "not saved"}) is None


@pytest.mark.parametrize("steps", [[], [{}], [{"id": 1, "description": "save", "detect": ""}]])
def test_invalid_legacy_procedure_rejected(tmp_path, steps):
    path = tmp_path / "procedure.json"
    path.write_text(json.dumps({"steps": steps}))
    with pytest.raises(ValueError):
        Procedure(str(path))


def test_mobile_procedure_requires_unique_explicit_predicates(tmp_path):
    path = tmp_path / "procedure.json"
    path.write_text(json.dumps({"steps": [{"id": "save", "expected": {"text": ""}}]}))
    with pytest.raises(ValueError):
        load_procedure(str(path))
