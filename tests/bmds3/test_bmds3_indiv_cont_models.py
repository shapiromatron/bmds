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
        (continuous.ExponentialM3, [394.278, 257.847, 877.507], 116.4),
        (continuous.ExponentialM5, [256.441, 106.008, -9999.0], 111.5),
        (continuous.Power, [386.1, 247.22, 878.062], 114.3),
        (continuous.Hill, [178.395, 105.924, -9999.0], 111.5),
        (continuous.Linear, [386.097, 247.215, 878.086], 114.3),
        (continuous.Polynomial, [376.489, 246.938, 384.282], 114.3),
    ]:
        result = Model(cidataset).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 3).tolist()}, {round(result.fit.aic, 1)}),"
        # print(res)
        assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__
