import copy
import json
from pathlib import Path

import pytest

from clef_quickstart import load_request, validate_request

EXAMPLE = Path(__file__).parents[1] / "examples" / "incident.json"


def test_example_is_valid():
    request = load_request(EXAMPLE)
    assert request["model"] == "clef-flash"
    assert set(request["questions"]) == {"department", "urgency", "outage"}


@pytest.mark.parametrize("missing", ["model", "state", "questions"])
def test_required_fields(missing):
    request = json.loads(EXAMPLE.read_text())
    request.pop(missing)
    with pytest.raises(ValueError):
        validate_request(request)

def test_choice_needs_two_options():
    request = json.loads(EXAMPLE.read_text())
    request = copy.deepcopy(request)
    request["questions"]["department"]["criteria"] = {"technical": "Bugs"}
    with pytest.raises(ValueError, match="at least two"):
        validate_request(request)
