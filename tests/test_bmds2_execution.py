import inspect
import os

import numpy as np
import pytest
from simple_settings import settings
from simple_settings.utils import settings_stub

import bmds


def test_executable_path():

    parents = (bmds.models.Dichotomous, bmds.models.DichotomousCancer, bmds.models.Continuous)

    for name, obj in inspect.getmembers(bmds):
        if inspect.isclass(obj):
            if obj not in parents and issubclass(obj, parents):
                exe = obj.get_exe_path()
                assert os.path.exists(exe)


@pytest.mark.vcr()
def test_default_execution(cdataset, ddataset, cidataset):
    # All models execute given valid inputs

    # CONTINUOUS
    session = bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_default_models()
    assert len(session.models) == 10
    session.execute()
    for model in session.models:

        # check correct completion
        assert model.output_created is True
        assert len(model.outfile) > 0
        assert model.execution_duration >= 0
        assert "execution_end_time" in model.output

    # check BMDU are created (v2.7)
    actual = np.array([m.output["BMDU"] for m in session.models])
    expected = np.array(
        [112.485, 112.485, 112.485, 112.485, 112.485, 79.6883, 87.6812, 87.6812, 65.4371, 81.3828]
    )
    assert np.isclose(actual, expected).all()

    # CONTINUOUS INDIVIDUAL
    session = bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS_INDIVIDUAL, dataset=cidataset)
    session.add_default_models()
    assert len(session.models) == 12
    session.execute()
    for model in session.models:

        # check correct completion
        assert model.output_created is True
        assert len(model.outfile) > 0
        assert model.execution_duration >= 0
        assert "execution_end_time" in model.output

    # check BMDU are created (v2.7)
    actual = np.array([m.output["BMDU"] for m in session.models])
    expected = np.array(
        [
            878.068,
            878.068,
            878.068,
            878.068,
            878.068,
            878.068,
            878.068,
            2500000000.0,
            880.338,
            880.338,
            5000000.0,
            5000000.0,
        ]
    )
    assert np.isclose(actual, expected).all()

    # DICHOTOMOUS
    session = bmds.BMDS.version("BMDS270", bmds.constants.DICHOTOMOUS, dataset=ddataset)
    session.add_default_models()
    assert len(session.models) == 10
    session.execute()
    for model in session.models:

        # check correct completion
        assert model.output_created is True
        assert len(model.outfile) > 0
        assert model.execution_duration >= 0
        assert "execution_end_time" in model.output

    # check BMDU are created (v2.7)
    actual = np.array([m.output["BMDU"] for m in session.models])
    expected = np.array(
        [23.6386, 29.105, 23.2359, 29.4044, 22.8424, 24.6612, 26.3032, 28.1727, 29.1446, -999]
    )
    assert np.isclose(actual, expected).all()

    # DICHOTOMOUS CANCER
    session = bmds.BMDS.version("BMDS270", bmds.constants.DICHOTOMOUS_CANCER, dataset=ddataset)
    session.add_default_models()
    assert len(session.models) == 3
    session.execute()
    for model in session.models:

        # check correct completion
        assert model.output_created is True
        assert len(model.outfile) > 0
        assert model.execution_duration >= 0
        assert "execution_end_time" in model.output

    # check BMDU are created (v2.7)
    actual = np.array([m.output["BMDU"] for m in session.models])
    expected = np.array([22.8424, 24.6612, 26.3032])
    assert np.isclose(actual, expected).all()

    # confirm cancer slope factor exists
    actual = np.array([m.output["CSF"] for m in session.models])
    expected = np.array([0.0129348, 0.0108767, 0.0108578])
    assert np.isclose(actual, expected).all()


@pytest.mark.vcr()
def test_parameter_settings(cdataset):
    # assert to settings are used
    session = bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_model(bmds.constants.M_Polynomial)
    session.add_model(
        bmds.constants.M_Polynomial, settings={"constant_variance": 1, "degree_poly": 3}
    )

    session.execute()
    model1 = session.models[0]
    model2 = session.models[1]

    assert model1.output_created is True and model2.output_created is True

    # check model-variance setting
    assert "The variance is to be modeled" in model1.outfile
    assert "rho" in model1.output["parameters"]
    assert "A constant variance model is fit" in model2.outfile
    assert "rho" not in model2.output["parameters"]

    # check degree_poly override setting
    assert "beta_3" not in model1.output["parameters"]
    assert model2.output["parameters"]["beta_3"]["estimate"] == 0.0


@pytest.mark.vcr()
def test_tiny_datasets():
    # Observation # < parameters # for Hill model.
    # Make sure this doesn't break execution or recommendation.
    ds = bmds.ContinuousDataset(
        doses=[0.0, 4.4, 46.0], ns=[24, 16, 16], means=[62.3, 40.6, 39.9], stdevs=[8.4, 3.4, 4.3]
    )
    session = bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS, dataset=ds)
    session.add_model(bmds.constants.M_Hill)
    session.add_model(bmds.constants.M_ExponentialM5)
    session.execute()
    session.recommend()
    assert session.recommended_model_index is None

    ds = bmds.DichotomousDataset(doses=[0.0, 4.4, 46.0], ns=[16, 16, 16], incidences=[2, 5, 9])
    session = bmds.BMDS.version("BMDS270", bmds.constants.DICHOTOMOUS, dataset=ds)
    session.add_model(bmds.constants.M_DichotomousHill)
    session.execute()
    session.recommend()
    assert session.recommended_model_index is None


@pytest.mark.vcr()
def test_continuous_restrictions(cdataset):
    session = bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_model(bmds.constants.M_Power)
    session.add_model(bmds.constants.M_Power, settings={"restrict_power": 0})
    session.add_model(bmds.constants.M_Hill)
    session.add_model(bmds.constants.M_Hill, settings={"restrict_n": 0})

    session.execute()
    power1 = session.models[0]
    power2 = session.models[1]
    hill1 = session.models[2]
    hill2 = session.models[3]

    for m in session.models:
        assert m.output_created is True

    assert "The power is restricted to be greater than or equal to 1" in power1.outfile  # noqa
    assert "The power is not restricted" in power2.outfile
    assert "Power parameter restricted to be greater than 1" in hill1.outfile
    assert "Power parameter is not restricted" in hill2.outfile


@pytest.mark.vcr()
def test_dichotomous_restrictions(ddataset):
    session = bmds.BMDS.version("BMDS270", bmds.constants.DICHOTOMOUS, dataset=ddataset)
    for model in session.model_options:
        session.add_model(bmds.constants.M_Weibull)
        session.add_model(bmds.constants.M_Weibull, settings={"restrict_power": 0})

    session.execute()
    weibull1 = session.models[0]
    weibull2 = session.models[1]

    for m in session.models:
        assert m.output_created is True

    assert "Power parameter is restricted as power >= 1" in weibull1.outfile
    assert "Power parameter is not restricted" in weibull2.outfile


def test_can_be_executed(bad_cdataset):
    # ensure exit-early function properly detects which models can and
    # cannot be executed, based on dataset size.

    assert bad_cdataset.num_dose_groups == 3

    model = bmds.models.Power_217(bad_cdataset)
    assert model.can_be_executed is True

    model = bmds.models.Exponential_M5_19(bad_cdataset)
    assert model.can_be_executed is False


@pytest.mark.vcr()
def test_bad_datasets(bad_cdataset, bad_ddataset):
    # ensure library doesn't fail with a terrible dataset that should never
    # be executed in the first place (which causes BMDS to throw NaN)

    session = bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS, dataset=bad_cdataset)
    session.add_default_models()
    session.execute()
    session.recommend()
    halted = [model.execution_halted for model in session.models]
    assert halted == [False] * 8
    assert session.recommended_model_index is None

    # gross; if we're on windows and actually existing then make timeout realistic; else we may be
    # using pytest vcr and then make it really fast
    timeout = 1e-5
    with settings_stub(BMDS_MODEL_TIMEOUT_SECONDS=timeout):
        # works in later versions; fix to this version
        BMDSv2601 = bmds.BMDS.versions["BMDS2601"]
        session = BMDSv2601(bmds.constants.DICHOTOMOUS, dataset=bad_ddataset)
        session.add_model(bmds.constants.M_Gamma)
        session.execute()
        session.recommend()
        total_time = session.models[0].execution_duration
        timeout = settings.BMDS_MODEL_TIMEOUT_SECONDS

        assert session.recommended_model_index is None
        assert session.models[0].execution_halted is True
        assert np.isclose(total_time, timeout) or total_time > timeout


@pytest.mark.vcr()
def test_execute_with_dosedrop(ddataset_requires_dose_drop):
    session = bmds.BMDS.version(
        "BMDS270", bmds.constants.DICHOTOMOUS, dataset=ddataset_requires_dose_drop
    )
    session.add_model(bmds.constants.M_Logistic)
    session.execute_and_recommend(drop_doses=True)

    assert session.recommended_model_index == 0
    assert session.doses_dropped == 1
    assert len(session.dataset.ns) == len(session.original_dataset.ns) - 1


@pytest.mark.vcr()
def test_large_dataset():
    # N is so large that the residual table cannot be parsed correctly
    dataset = bmds.ContinuousDataset(
        doses=[0, 1, 10, 50, 100],
        ns=[1244339, 39153, 58064, 58307, 56613],
        means=[156.70, 159.00, 156.07, 161.71, 159.78],
        stdevs=[72.46, 74.47, 73.60, 84.24, 81.89],
    )
    session = bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS, dataset=dataset)
    session.add_model(bmds.constants.M_Power)

    with pytest.raises(ValueError) as err:
        session.execute()

    assert "Fit table could not be parsed" in str(err)
