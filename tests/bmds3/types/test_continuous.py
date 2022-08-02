import pytest

from bmds.bmds3.models import continuous

from ..run3 import RunBmds3


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
class TestContinuousGof:
    def test_collapse(self, cdataset, cidataset):
        # goodness of fit should collapse into non-zero fields

        # continuous summary data already collapsed; no change in length of table
        model = continuous.Power(cdataset)
        res = model.execute()
        assert res.gof.n() == cdataset.num_dose_groups == 5
        assert res.gof.n() == len(cdataset.doses) == 5

        # continuous individual summary data collapses appropriately
        model = continuous.Power(cidataset)
        res = model.execute()
        assert res.gof.n() == cidataset.num_dose_groups == 7
        assert res.gof.n() == len(cidataset.doses)
        assert res.gof.n() < len(cidataset.individual_doses)
        assert res.gof.n() == len(set(cidataset.individual_doses))


class TestContinuousParameters:
    def test_exp3(self, cdataset):
        model = continuous.ExponentialM3(cdataset)
        res = model.execute()
        # param names for prior are as expected
        assert model.get_param_names() == ["a", "b", "c", "d", "rho"]
        # but outputs have been shifted
        assert res.parameters.names == ["a", "b", "d", "rho", "NULL"]
