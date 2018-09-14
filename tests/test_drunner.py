import json
import pytest
import sys

import bmds

from .fixtures import *  # noqa


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows")
def test_drunner(cdataset):
    payload = bmds.monkeypatch._get_payload(
        [bmds.models.Linear_220(cdataset), bmds.models.Power_218(cdataset)]
    )
    inputs = json.loads(payload)["inputs"]
    runner = bmds.BatchDfileRunner(inputs)
    outputs = runner.execute()
    assert len(outputs) == 2
    assert "BMD =        99.9419" in outputs[0]["output"]
    assert "BMD = 99.9419" in outputs[1]["output"]
