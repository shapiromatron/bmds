import json
import os

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


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_dichotomous_ma_session(dichds):
    session = bmds.session.Bmds330(dataset=dichds)
    session.add_default_bayesian_models()
    session.execute()
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))
