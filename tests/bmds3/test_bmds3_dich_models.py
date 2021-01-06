import json
import os

import numpy as np
import pytest

import bmds
from bmds.bmds3.models import dichotomous

# TODO remove this restriction
should_run = os.getenv("CI") is None
skip_reason = "DLLs not present on CI"


@pytest.fixture
def dichds():
    return bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_dichotomous_models(dichds):
    # compare bmd, bmdl, bmdu, aic values
    for Model, expected in [
        (dichotomous.Logistic, [69.583, 61.194, 77.945, 363.957]),
        (dichotomous.LogLogistic, [68.163, 59.795, 75.998, 364.982]),
        (dichotomous.Probit, [66.874, 58.335, 75.368, 362.057]),
        (dichotomous.LogProbit, [66.138, 58.681, 73.153, 366.271]),
        (dichotomous.Gamma, [66.037, 57.639, 73.715, 363.607]),
        (dichotomous.QuantalLinear, [17.679, 15.645, 20.062, 425.594]),
        (dichotomous.Weibull, [64.242, 55.219, 72.814, 362.400]),
        (dichotomous.DichotomousHill, [68.173, 59.795, 75.998, 366.982]),
        (dichotomous.Multistage, [64.127, 52.552, 76.220, 366.384]),
    ]:
        model = Model(dichds)
        result = model.execute()
        actual = [result.bmd, result.bmdl, result.bmdu, result.aic]
        # for regenerating expected: `print(Model.__name__, np.round(actual, 3).tolist())`
        assert np.isclose(np.array(actual), np.array(expected), atol=1e-2).all(), Model.__name__


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_dichotomous_session(dichds):
    session = bmds.session.Bmds330(dataset=dichds)
    session.add_default_models()
    session.execute()
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))
