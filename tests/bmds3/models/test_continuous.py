import json

import pytest

import bmds
from bmds import constants
from bmds.bmds3.constants import BMDS_BLANK_VALUE, DistType, PriorClass
from bmds.bmds3.models import continuous
from bmds.bmds3.types.continuous import ContinuousModelSettings

from ..run3 import RunBmds3


class TestPriorOverrides:
    def test_hill(self, cdataset2):
        for settings, priors in [
            ({"is_increasing": True}, (0, 100, 1.2)),
            ({"is_increasing": False}, (-100, 0, 1.2)),
            ({"disttype": DistType.normal}, (0, 100, 1.2)),
            ({"disttype": DistType.normal_ncv}, (0, 100, 0.1823)),
        ]:
            model = continuous.Hill(cdataset2, settings)
            v = model.settings.priors.get_prior("v")
            n = model.settings.priors.get_prior("n")
            assert v.min_value == priors[0], settings
            assert v.max_value == priors[1], settings
            assert n.stdev == priors[2], settings

    def test_poly(self, cdataset2, negative_cdataset):
        for settings, priors in [
            # fmt: off
            ({"is_increasing": True}, (0, 1e6)),
            ({"is_increasing": False}, (-1e6, 0)),
            ({"is_increasing": True, "disttype": DistType.normal_ncv}, (0, 18)),
            ({"is_increasing": False, "disttype": DistType.normal_ncv}, (-18, 0)),
            ({"priors": PriorClass.frequentist_unrestricted, "is_increasing": True}, (-1e6, 1e6)),
            ({"priors": PriorClass.frequentist_unrestricted, "is_increasing": False}, (-1e6, 1e6)),
            ({"priors": PriorClass.frequentist_unrestricted, "is_increasing": True, "disttype": DistType.normal_ncv}, (-18, 18)),
            ({"priors": PriorClass.frequentist_unrestricted, "is_increasing": False, "disttype": DistType.normal_ncv}, (-18, 18)),
            # fmt: on
        ]:
            model = continuous.Polynomial(cdataset2, settings)
            betaN = model.settings.priors.get_prior("betaN")
            assert betaN.min_value == priors[0], settings
            assert betaN.max_value == priors[1], settings

    def test_power(self, cdataset2):
        for settings, priors in [
            ({}, [0.1, -100, 100]),
            ({"disttype": DistType.normal_ncv}, [1, -10000, 10000]),
        ]:
            model = continuous.Power(cdataset2, settings)
            g = model.settings.priors.get_prior("g")
            v = model.settings.priors.get_prior("v")
            assert g.stdev == priors[0]
            assert v.min_value == priors[1]
            assert v.max_value == priors[2]


class TestBmdModelContinuous:
    def test_get_param_names(self, cdataset2):
        # test normal model case
        for m in [
            continuous.Power(dataset=cdataset2),
            continuous.Power(dataset=cdataset2, settings=dict(disttype=DistType.normal)),
            continuous.Power(dataset=cdataset2, settings=dict(disttype=DistType.log_normal)),
        ]:
            assert m.get_param_names() == ["g", "v", "n", "alpha"]
        m = continuous.Power(dataset=cdataset2, settings=dict(disttype=DistType.normal_ncv))
        assert m.get_param_names() == ["g", "v", "n", "rho", "alpha"]

        # test polynomial
        model = continuous.Linear(dataset=cdataset2)
        assert model.get_param_names() == ["g", "b1", "alpha"]
        model = continuous.Polynomial(dataset=cdataset2)
        assert model.get_param_names() == ["g", "b1", "b2", "alpha"]
        model = continuous.Polynomial(dataset=cdataset2, settings=dict(degree=3))
        assert model.get_param_names() == ["g", "b1", "b2", "b3", "alpha"]
        model = continuous.Polynomial(
            dataset=cdataset2, settings=dict(degree=3, disttype=DistType.normal_ncv)
        )
        assert model.get_param_names() == ["g", "b1", "b2", "b3", "rho", "alpha"]

    @pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
    def test_report(self, cdataset2):
        model = continuous.Hill(dataset=cdataset2)
        text = model.text()
        assert "Hill" in text
        assert "Model has not successfully executed; no results available." in text

        model.execute()
        text = model.text()
        assert "Hill" in text
        assert "Goodness of Fit:" in text

    def test_default_prior_class(self, cdataset2):
        for Model, prior_class in [
            (continuous.ExponentialM3, PriorClass.frequentist_restricted),
            (continuous.ExponentialM5, PriorClass.frequentist_restricted),
            (continuous.Power, PriorClass.frequentist_restricted),
            (continuous.Hill, PriorClass.frequentist_restricted),
            (continuous.Linear, PriorClass.frequentist_unrestricted),
            (continuous.Polynomial, PriorClass.frequentist_restricted),
        ]:
            assert Model(cdataset2).settings.priors.prior_class is prior_class

    @pytest.mark.mpl_image_compare
    @pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
    def test_bmds3_continuous_plot(self, cdataset2):
        model = continuous.Hill(dataset=cdataset2)
        model.execute()
        return model.plot()


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_increasing(cdataset2):
    """
    Basic tests to ensure AIC and BMD values are successfully created and stable for all model classes
    """
    # test increasing means dataset
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [53, 50, 55], 3186),
        (continuous.ExponentialM5, [26, 25, 28], 3072),
        (continuous.Power, [26, 24, 30], 3070),
        (continuous.Hill, [28, 25, 32], 3072),
        (continuous.Linear, [26, 24, 28], 3068),
        (continuous.Polynomial, [26, 24, 29], 3070),
    ]:
        result = Model(cdataset2).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 0).astype(int).tolist()}, {round(result.fit.aic)}),"
        # print(res)
        assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_decreasing(negative_cdataset):
    # test decreasing means dataset
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [BMDS_BLANK_VALUE, BMDS_BLANK_VALUE, BMDS_BLANK_VALUE], 4298),
        (continuous.ExponentialM5, [BMDS_BLANK_VALUE, BMDS_BLANK_VALUE, BMDS_BLANK_VALUE], 4300),
        (continuous.Power, [56, 50, 64], 3080),
        (continuous.Hill, [63, 56, 70], 3086),
        (continuous.Linear, [35, 33, 38], 3117),
        (continuous.Polynomial, [52, 47, 60], 3077),
    ]:
        model = Model(negative_cdataset)
        result = model.execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 0).astype(int).tolist()}, {round(result.fit.aic)}),"
        # print(res)
        assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_variance(cdataset2):
    model = continuous.Power(cdataset2, dict(disttype=DistType.normal))
    result = model.execute()
    actual = [result.bmd, result.bmdl, result.bmdu]
    # print(f"{actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f}")
    assert model.settings.disttype is DistType.normal
    assert pytest.approx(actual, rel=0.05) == [25.9, 24.4, 29.8]
    assert result.parameters.names == ["g", "v", "n", "alpha"]

    model = continuous.Power(cdataset2, dict(disttype=DistType.normal_ncv))
    result = model.execute()
    actual = [result.bmd, result.bmdl, result.bmdu]
    # print(f"{actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f}")
    assert model.settings.disttype is DistType.normal_ncv
    assert result.parameters.names == ["g", "v", "n", "rho", "alpha"]
    assert pytest.approx(actual, rel=0.05) == [14.6, 13.1, 17.3]

    # only Exp can be used
    model = continuous.ExponentialM5(cdataset2, dict(disttype=DistType.log_normal))
    result = model.execute()
    actual = [result.bmd, result.bmdl, result.bmdu]
    # print(f"{actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f}")
    assert model.settings.disttype is DistType.log_normal
    assert pytest.approx(actual, rel=0.05) == [10.3, 9.71, 11.1]
    assert result.parameters.names == ["a", "b", "c", "d", "log-alpha"]


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_continuous_polynomial(cdataset2):
    # compare bmd, bmdl, bmdu, aic values
    for degree, bmd_values, aic in [
        (1, [25.84, 24.358, 27.529], 3067.8),
        (2, [25.984, 24.328, 28.642], 3069.8),
        (3, [26.803, 24.259, 28.908], 3070.3),
        (4, [26.137, 24.336, 28.713], 3069.9),
    ]:
        settings = ContinuousModelSettings(degree=degree)
        result = continuous.Polynomial(cdataset2, settings).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"({degree}, {np.round(actual, 3).tolist()}, {round(result.fit.aic, 1)}),"
        # print(res)
        assert pytest.approx(actual, rel=0.05) == bmd_values, degree
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, degree


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_continuous_session(cdataset2):
    session = bmds.session.Bmds330(dataset=cdataset2)
    session.add_default_models()
    session.execute()
    for model in session.models:
        model.results = model.execute()
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_increasing_lognormal(cdataset2):
    session = bmds.session.Bmds330(dataset=cdataset2)
    settings = dict(disttype=DistType.log_normal)
    for model in (constants.M_ExponentialM3, constants.M_ExponentialM5):
        session.add_model(model, settings)
    session.execute()
    for model in session.models:
        assert model.results.has_completed is True
        assert model.results.bmd != BMDS_BLANK_VALUE

    session = bmds.session.Bmds330(dataset=cdataset2)
    settings = dict(disttype=DistType.log_normal)
    for model in (constants.M_Hill, constants.M_Power, constants.M_Polynomial):
        session.add_model(model, settings)
    session.execute()
    for model in session.models:
        assert model.results.has_completed is False
        assert model.results.bmd == BMDS_BLANK_VALUE


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_decreasing_lognormal(negative_cdataset):
    session = bmds.session.Bmds330(dataset=negative_cdataset)
    settings = dict(disttype=DistType.log_normal)
    for model in (constants.M_ExponentialM3, constants.M_ExponentialM5):
        session.add_model(model, settings)
    session.execute()
    for model in session.models:
        assert model.results.has_completed is True
        assert model.results.bmd == pytest.approx(100, rel=0.05)

    session = bmds.session.Bmds330(dataset=negative_cdataset)
    settings = dict(disttype=DistType.log_normal)
    for model in (constants.M_Hill, constants.M_Power, constants.M_Polynomial):
        session.add_model(model, settings)
    session.execute()
    for model in session.models:
        assert model.results.has_completed is False
        assert model.results.bmd == BMDS_BLANK_VALUE