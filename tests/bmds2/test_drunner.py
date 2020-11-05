import json
import sys

import pytest

from bmds.bmds2 import drunner, models, remote


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows")
def test_drunner(cdataset):
    payload = remote._get_payload([models.Linear_221(cdataset), models.Power_219(cdataset)])
    inputs = json.loads(payload)["inputs"]
    runner = drunner.BatchDfileRunner(inputs)
    outputs = runner.execute()
    assert len(outputs) == 2
    assert "BMD =        99.9419" in outputs[0]["output"]
    assert "BMD = 99.9419" in outputs[1]["output"]
