import json
import pytest
import sys

import bmds

from .fixtures import *  # noqa


@pytest.mark.skipif(sys.platform != 'win32', reason='requires Windows')
def test_drunner(cdataset):
    model = bmds.models.Power_218(cdataset)
    payload = bmds.monkeypatch._get_payload([model])
    inputs = json.loads(payload['inputs'])
    runner = bmds.BatchDfileRunner(inputs)
    outputs = runner.execute()
    assert 'BMD = 99.9419' in outputs[0]['output']
