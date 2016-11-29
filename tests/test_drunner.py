import pytest
import json

import bmds


@pytest.fixture
def inputs():
    cdataset = bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[111, 142, 143, 93, 42],
        responses=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159])
    model = bmds.models.Power_218(cdataset)
    payload = bmds.get_payload([model])
    inputs = json.loads(payload['inputs'])
    return inputs


def test_drunner(inputs):
    runner = bmds.BatchDfileRunner(inputs)
    outputs = runner.execute()
    assert outputs[0]['output_created'] is True
