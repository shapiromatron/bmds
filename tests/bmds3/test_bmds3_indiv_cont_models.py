# import numpy as np
import pytest
from run3 import RunBmds3

from bmds.bmds3.models import continuous


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_bmds3_increasing(cidataset):
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [394, 257, 877], 114),
        (continuous.ExponentialM5, [257, 104, -9999], 111),
        (continuous.Power, [407, 237, 415], 117),
        (continuous.Hill, [177, 105, -9999.0], 111),
        (continuous.Linear, [386, 247, 878], 114),
        (continuous.Polynomial, [387, 246, 878], 114),
    ]:
        result = Model(cidataset).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 3).tolist()}, {round(result.fit.aic, 1)}),"
        # print(res)
        assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__
