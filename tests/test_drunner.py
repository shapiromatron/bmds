import json
import pytest
import sys

import bmds

from .fixtures import *  # noqa


@pytest.fixture
def inputs(cdataset):
    model = bmds.models.Power_218(cdataset)
    payload = bmds.monkeypatch._get_payload([model])
    inputs = json.loads(payload['inputs'])
    return inputs


@pytest.mark.skipif(sys.platform != 'win32', reason="requires Windows")
def test_drunner(inputs):
    runner = bmds.BatchDfileRunner(inputs)
    outputs = runner.execute()
    assert outputs[0]['output_created'] is True
