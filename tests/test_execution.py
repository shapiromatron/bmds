import numpy as np
import os
import inspect

import bmds
from bmds import models
from simple_settings import settings
from simple_settings.utils import settings_stub
import pytest

from .fixtures import *  # noqa


def test_executable_path():

    parents = (bmds.models.Dichotomous, bmds.models.DichotomousCancer, bmds.models.Continuous)

    for name, obj in inspect.getmembers(bmds):
        if inspect.isclass(obj):
            if obj not in parents and issubclass(obj, parents):
                exe = obj.get_exe_path()
                assert os.path.exists(exe)


def test_default_execution(cdataset, ddataset, cidataset):
    # All models execute given valid inputs

    # CONTINUOUS
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
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
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS_INDIVIDUAL, dataset=cidataset)
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
    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=ddataset)
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
    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS_CANCER, dataset=ddataset)
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


def test_parameter_overrides(cdataset):
    # assert to overrides are used
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_model(bmds.constants.M_Polynomial)
    session.add_model(
        bmds.constants.M_Polynomial, overrides={"constant_variance": 1, "degree_poly": 3}
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


def test_tiny_datasets():
    # Observation # < parameters # for Hill model.
    # Make sure this doesn't break execution or recommendation.
    ds = bmds.ContinuousDataset(
        doses=[0.0, 4.4, 46.0], ns=[24, 16, 16], means=[62.3, 40.6, 39.9], stdevs=[8.4, 3.4, 4.3]
    )
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=ds)
    session.add_model(bmds.constants.M_Hill)
    session.add_model(bmds.constants.M_ExponentialM5)
    session.execute()
    session.recommend()
    assert session.recommended_model_index is None

    ds = bmds.DichotomousDataset(doses=[0.0, 4.4, 46.0], ns=[16, 16, 16], incidences=[2, 5, 9])
    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=ds)
    session.add_model(bmds.constants.M_DichotomousHill)
    session.execute()
    session.recommend()
    assert session.recommended_model_index is None


def test_continuous_restrictions(cdataset):
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_model(bmds.constants.M_Power)
    session.add_model(bmds.constants.M_Power, overrides={"restrict_power": 0})
    session.add_model(bmds.constants.M_Hill)
    session.add_model(bmds.constants.M_Hill, overrides={"restrict_n": 0})

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


def test_dichotomous_restrictions(ddataset):
    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=ddataset)
    for model in session.model_options:
        session.add_model(bmds.constants.M_Weibull)
        session.add_model(bmds.constants.M_Weibull, overrides={"restrict_power": 0})

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


def test_bad_datasets(bad_cdataset, bad_ddataset):
    # ensure library doesn't fail with a terrible dataset that should never
    # be executed in the first place (which causes BMDS to throw NaN)

    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=bad_cdataset)
    session.add_default_models()
    session.execute()
    session.recommend()
    assert session.recommended_model_index is None

    with settings_stub(BMDS_MODEL_TIMEOUT_SECONDS=3):
        # works in later versions; fix to this version
        BMDSv2601 = bmds.BMDS.versions["BMDS2601"]
        session = BMDSv2601(bmds.constants.DICHOTOMOUS, dataset=bad_ddataset)
        session.add_default_models()
        session.execute()
        session.recommend()
        assert session.recommended_model_index is None

        # assert that the execution_halted flag is appropriately set
        halted = [model.execution_halted for model in session.models]
        str_halted = (
            "[False, False, False, False, False, False, False, True, False, False]"  # noqa
        )
        assert halted[7] is True and session.models[7].name == "Gamma"
        assert str(halted) == str_halted
        total_time = session.models[7].execution_duration
        timeout = settings.BMDS_MODEL_TIMEOUT_SECONDS
        assert np.isclose(total_time, timeout) or total_time > timeout


@pytest.mark.skip(reason="timeout failure is unreliable")
def test_timeout():
    # confirm that timeout setting works as expected; slow dataset provided
    # by CEBS team on 2017-11-12. Only slow in BMDS version 2.6; fixed in 2.7.

    dataset = bmds.ContinuousIndividualDataset(
        doses=[
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.000316,
            0.000316,
            0.000316,
            0.000316,
            0.001,
            0.001,
            0.001,
            0.001,
            0.00316,
            0.00316,
            0.00316,
            0.00316,
            0.01,
            0.01,
            0.01,
            0.01,
            0.0316,
            0.0316,
            0.0316,
            0.0316,
            0.1,
            0.1,
            0.1,
            0.1,
            0.316,
            0.316,
            0.316,
            0.316,
            1,
            1,
            1,
            1,
            3.16,
            3.16,
            3.16,
            3.16,
            10,
            10,
            10,
            10,
        ],
        responses=[
            2289000,
            2047000,
            2108000,
            2148000,
            2325000,
            2014000,
            2173000,
            2261000,
            2024000,
            2272000,
            1742000,
            1970000,
            1850000,
            1030000,
            2074000,
            2159000,
            2348000,
            2270000,
            2238000,
            2082000,
            1894000,
            1829000,
            2181000,
            2044000,
            2438000,
            2264000,
            2303000,
            2316000,
            2061000,
            2165000,
            2310000,
            2294000,
            2550000,
            2076000,
            2284000,
            2249000,
            2308000,
            2096000,
            2347000,
            2340000,
            2170000,
            1916000,
            2858000,
            2448000,
            2648000,
            2226000,
            1164000,
            1283000,
            1278000,
            1577000,
            40305,
            36227,
            27300,
            21531,
        ],
    )

    with settings_stub(BMDS_MODEL_TIMEOUT_SECONDS=3):
        model = models.Hill_217(dataset)
        model.execute()
        assert model.has_successfully_executed is False

    with settings_stub(BMDS_MODEL_TIMEOUT_SECONDS=10):
        model = models.Hill_217(dataset)
        model.execute()
        assert model.has_successfully_executed is True


def test_execute_with_dosedrop(ddataset_requires_dose_drop):
    session = bmds.BMDS.latest_version(
        bmds.constants.DICHOTOMOUS, dataset=ddataset_requires_dose_drop
    )
    session.add_model(bmds.constants.M_Logistic)
    session.execute_and_recommend(drop_doses=True)

    assert session.recommended_model_index == 0
    assert session.doses_dropped == 1
    assert len(session.dataset.ns) == len(session.original_dataset.ns) - 1


def test_large_dataset():
    # N is so large that the residual table cannot be parsed correctly
    dataset = bmds.ContinuousDataset(
        doses=[0, 1, 10, 50, 100],
        ns=[1244339, 39153, 58064, 58307, 56613],
        means=[156.70, 159.00, 156.07, 161.71, 159.78],
        stdevs=[72.46, 74.47, 73.60, 84.24, 81.89],
    )
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=dataset)
    session.add_model(bmds.constants.M_Power)

    with pytest.raises(ValueError) as err:
        session.execute_and_recommend(drop_doses=True)

    assert "Fit table could not be parsed" in str(err)
