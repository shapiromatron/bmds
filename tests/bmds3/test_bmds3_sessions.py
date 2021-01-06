import json
import os
from pathlib import Path

import pytest

import bmds

# TODO remove this restriction
should_run = os.getenv("CI") is None
skip_reason = "DLLs not present on CI"


@pytest.fixture
def dichds():
    return bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )


def _format_serialized(fn: Path):
    data = json.loads(fn.read_text())
    data["version"]["numeric"] = tuple(data["version"]["numeric"])  # cast list -> tuple
    return data


@pytest.mark.skipif(not should_run, reason=skip_reason)
class TestBmds330:
    def test_serialization(self, dichds, data_path):
        # make sure serialize looks correct

        session1 = bmds.session.Bmds330(dataset=dichds)
        session1.add_default_models()
        session1.execute()

        serialized = session1.serialize()
        expected = _format_serialized(fn=data_path / "bmds3_session_serialization.json")

        assert serialized.dict() == expected

        # make sure we get the correct class back
        session2 = serialized.deserialize()
        assert isinstance(session2, bmds.session.Bmds330)

        # make sure we get the same result back after deserializing
        assert session1.serialize().dict() == session2.serialize().dict()
