import json
import os
from math import nan

import numpy as np
import pytest

import bmds
from bmds.bmds3.models import continuous
from bmds.bmds3.types.continuous import ContinuousModelSettings

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
        (continuous.ExponentialM2, [-9999.0, -9999.0, -9999.0, -4.0]),
        (continuous.ExponentialM3, [20.220, 19.211, 21.845, 3368.294]),
        (continuous.ExponentialM4, [-9999.0, -9999.0, -9999.0, -4.0]),
        (continuous.ExponentialM5, [31.270, 29.849, 34.013, 3073.712]),
        (continuous.Power, [-9999.0, -9999.0, -9999.0, 3067.833]),
        (continuous.Hill, [-9999.0, -9999.0, -9999.0, 3071.148]),
        (continuous.Linear, [25.839, 24.384, 27.457, 3065.833]),
    ]:
        result = Model(contds).execute()
        actual = [result.bmd, result.bmdl, result.bmdu, result.aic]
        # for regenerating expected: `print(Model.__name__, np.round(actual, 3).tolist())`
        assert np.isclose(np.array(actual), np.array(expected), atol=1e-1).all(), Model.__name__


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_continuous_polynomial(contds):
    # compare bmd, bmdl, bmdu, aic values
    for degree, expected in [
        (1, [25.839, 24.384, 27.457, 3065.833]),
        (2, [25.612, 24.191, 27.126, nan]),
        (3, [25.728, 25.364, 26.141, nan]),
    ]:
        settings = ContinuousModelSettings(degree=degree)
        model = continuous.Polynomial(contds, settings)
        result = model.execute()
        actual = [result.bmd, result.bmdl, result.bmdu, result.aic]
        # for regenerating expected: `print(Model.__name__, np.round(actual, 3).tolist())`
        assert np.isclose(np.array(actual), np.array(expected), atol=1e-2, equal_nan=True).all()


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_continuous_session(contds: bmds.ContinuousDataset):
    session = bmds.session.Bmds330(dataset=contds)
    session.add_default_models()
    session.execute()
    for model in session.models:
        model.results = model.execute(debug=True)
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))
