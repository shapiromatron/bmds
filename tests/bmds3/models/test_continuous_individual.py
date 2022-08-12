# import numpy as np
import pytest

from bmds.bmds3.models import continuous

from ..run3 import RunBmds3


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
class TestBmdModelContinuousIndividual:
    def test_bmds3_increasing(self, cidataset):
        for Model, bmd_values, aic in [
            (continuous.ExponentialM3, [394, 258, 881], 114),
            (continuous.ExponentialM5, [258, 104, -9999], 112),
            (continuous.Power, [386, 247, 878], 114),
            (continuous.Hill, [178, 106, -9999], 112),
            (continuous.Linear, [386, 247, 878], 114),
            (continuous.Polynomial, [386, 247, 878], 114),
        ]:
            result = Model(cidataset).execute()
            actual = [result.bmd, result.bmdl, result.bmdu]
            # for regenerating values
            # import numpy as np
            # res = f"(continuous.{Model.__name__}, {np.round(actual, 0).astype(int).tolist()}, {round(result.fit.aic)}),"
            # print(res)
            assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
            assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__

    @pytest.mark.mpl_image_compare
    def test_bmds3_continuous_individual_plot(self, cidataset):
        model = continuous.Power(dataset=cidataset)
        model.execute()
        return model.plot()
