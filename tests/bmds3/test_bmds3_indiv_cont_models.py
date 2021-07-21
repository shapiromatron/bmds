import os

# import numpy as np
import pytest

from bmds.bmds3.models import continuous

# TODO remove this restriction
should_run = os.getenv("CI") is None
skip_reason = "DLLs not present on CI"


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_increasing(cidataset):
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [540.704, 481.411, 688.252], 123.0),
        (continuous.ExponentialM5, [239.787, 110.911, 309.187], 113.6),
        (continuous.Power, [386.188, 261.844, 750.561], 114.3),
        (continuous.Hill, [150.698, 105.971, -9999.0], 113.6),
        (continuous.Linear, [385.464, 247.295, 878.057], 114.3),
        (continuous.Polynomial, [386.115, 247.29, 878.051], 114.3),
    ]:
        result = Model(cidataset).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 3).tolist()}, {round(result.fit.aic, 1)}),"
        # print(res)
        assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__
