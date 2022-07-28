# import numpy as np
import pytest
from run3 import RunBmds3

from bmds.bmds3.models import continuous


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_bmds3_increasing(cidataset):
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [471, 245, 481], 115),
        (continuous.ExponentialM5, [438, 428, -9999], 116),
        (continuous.Power, [466, 234, 1246], 117),
        (continuous.Hill, [178, 106, -9999], 112),
        (continuous.Linear, [466, 0, 1306], 117),
        (continuous.Polynomial, [466, 0, 1306], 117),
    ]:
        result = Model(cidataset).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        import numpy as np

        res = f"(continuous.{Model.__name__}, {np.round(actual, 0).astype(int).tolist()}, {round(result.fit.aic)}),"
        print(res)
        # assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
        # assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__
