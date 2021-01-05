import json
import os

import numpy as np
import pytest

import bmds
from bmds.bmds3.models import continuous

# TODO remove this restriction
should_run = os.getenv("CI") is None
skip_reason = "DLLs not present on CI"


@pytest.fixture
def contds():
    return bmds.ContinuousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[100, 100, 100, 100, 100],
        means=[10, 20, 30, 40, 50],
        stdevs=[3, 4, 5, 6, 7],
    )


@pytest.mark.skipif(not should_run, reason="TODO - figure out why this one randomly fails")
def test_bmds3_continuous_models(contds):
    # compare bmd, bmdl, bmdu, aic values
    for Model, expected in [
        (continuous.ExponentialM2, [-9999.0, -9999.0, -9999.0, -6.0]),
        (continuous.ExponentialM3, [-9999.0, -9999.0, -9999.0, 295.8651958579271]),
        (continuous.ExponentialM4, [-9999.0, -9999.0, -9999.0, -6.0]),
        (continuous.ExponentialM5, [-9999.0, -9999.0, -9999.0, 303.631]),
        (continuous.Power, [-9999.0, -9999.0, -9999.0, 301.627]),
        (continuous.Hill, [-9999.0, -9999.0, -9999.0, 303.629]),
        # (continuous.Polynomial, [64.242, 55.219, 72.814, 362.400]),
        # (continuous.Linear, [64.242, 55.219, 72.814, 362.400]),
    ]:
        result = Model(contds).execute()
        actual = [result.bmd, result.bmdl, result.bmdu, result.aic]
        # for regenerating expected: `print(Model.__name__, np.round(actual, 3).tolist())`
        assert np.isclose(np.array(actual), np.array(expected), atol=1e-1).all(), Model.__name__


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_continuous_session(contds: bmds.ContinuousDataset):
    session = bmds.session.BMDS_v330(dataset=contds)
    session.add_default_models()
    session.execute()
    for model in session.models:
        model.results = model.execute(debug=True)
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))
