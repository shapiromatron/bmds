from bmds.bmds3.constants import DistType
from bmds.bmds3.models import continuous


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
        """
        Edge case for exp3 - the dll expects a prior for the c parameter, but the
        returned output effectively drops the c array and shifts all other values down one.
        We check that the input and output values are shifted as required.
        """
        model = continuous.ExponentialM3(cdataset, settings=dict(disttype=DistType.normal))
        res = model.execute()
        # param names for prior are as expected
        assert model.get_param_names() == ["a", "b", "c", "d", "log-alpha"]
        # but outputs have been shifted
        assert res.parameters.names == ["a", "b", "d", "log-alpha"]

        model = continuous.ExponentialM3(cdataset, settings=dict(disttype=DistType.normal_ncv))
        res = model.execute()
        # param names for prior are as expected
        assert model.get_param_names() == ["a", "b", "c", "d", "rho", "log-alpha"]
        # but outputs have been shifted
        assert res.parameters.names == ["a", "b", "d", "rho", "log-alpha"]

        # confirm arrays all the same length after changes
        params = res.parameters
        n_params = len(params.names)
        for field in [
            "values",
            "se",
            "lower_ci",
            "upper_ci",
            "bounded",
            "prior_type",
            "prior_initial_value",
            "prior_stdev",
            "prior_min_value",
            "prior_max_value",
        ]:
            assert getattr(params, field).size == n_params
        assert params.cov.shape == (n_params, n_params)
