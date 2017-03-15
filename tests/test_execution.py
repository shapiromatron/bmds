import os
import inspect

import bmds

from .fixtures import *  # noqa


def test_executable_path():

    parents = (
        bmds.models.Dichotomous,
        bmds.models.DichotomousCancer,
        bmds.models.Continuous,
    )

    for name, obj in inspect.getmembers(bmds):
        if inspect.isclass(obj):
            if obj not in parents and issubclass(obj, parents):
                exe = obj.get_exe_path()
                print(obj.__name__, exe)
                assert os.path.exists(exe)


def test_default_execution(cdataset, ddataset, cidataset):
    # All models execute given valid inputs

    def _check_session(session, num_models):
        session.add_default_models()
        assert len(session.models) == num_models
        session.execute()
        for model in session.models:
            assert model.output_created is True
            assert len(model.outfile) > 0
            assert model.execution_duration > 0
            assert 'execution_end_time' in model.output

    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    _check_session(session, 10)

    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cidataset)
    _check_session(session, 12)

    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=ddataset)
    _check_session(session, 10)

    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS_CANCER, dataset=ddataset)
    _check_session(session, 3)


def test_parameter_overrides(cdataset):
    # assert to overrides are used
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_model(bmds.constants.M_Polynomial)
    session.add_model(bmds.constants.M_Polynomial,
                      overrides={'constant_variance': 1, 'degree_poly': 3})

    session.execute()
    model1 = session.models[0]
    model2 = session.models[1]

    assert model1.output_created is True and model2.output_created is True

    # check model-variance setting
    assert 'The variance is to be modeled' in model1.outfile
    assert 'rho' in model1.output['parameters']
    assert 'A constant variance model is fit' in model2.outfile
    assert 'rho' not in model2.output['parameters']

    # check degree_poly override setting
    assert 'beta_3' not in model1.output['parameters']
    assert model2.output['parameters']['beta_3']['estimate'] == 0.0


def test_tiny_datasets():
    # Observation # < parameters # for Hill model.
    # Make sure this doesn't break execution or recommendation.
    ds = bmds.ContinuousDataset(
        doses=[0.0, 4.4, 46.0],
        ns=[24, 16, 16],
        means=[62.3, 40.6, 39.9],
        stdevs=[8.4, 3.4, 4.3])
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=ds)
    session.add_model(bmds.constants.M_Hill)
    session.add_model(bmds.constants.M_ExponentialM5)
    session.execute()
    session.recommend()
    assert session.recommended_model_index is None

    ds = bmds.DichotomousDataset(
        doses=[0.0, 4.4, 46.0],
        ns=[16, 16, 16],
        incidences=[2, 5, 9])
    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=ds)
    session.add_model(bmds.constants.M_DichotomousHill)
    session.execute()
    session.recommend()
    assert session.recommended_model_index is None


def test_continuous_restrictions(cdataset):
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_model(bmds.constants.M_Power)
    session.add_model(bmds.constants.M_Power, overrides={'restrict_power': 0})
    session.add_model(bmds.constants.M_Hill)
    session.add_model(bmds.constants.M_Hill, overrides={'restrict_n': 0})

    session.execute()
    power1 = session.models[0]
    power2 = session.models[1]
    hill1 = session.models[2]
    hill2 = session.models[3]

    for m in session.models:
        assert m.output_created is True

    assert 'The power is restricted to be greater than or equal to 1' in power1.outfile
    assert 'The power is not restricted' in power2.outfile
    assert 'Power parameter restricted to be greater than 1' in hill1.outfile
    assert 'Power parameter is not restricted' in hill2.outfile


def test_dichotomous_restrictions(ddataset):
    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=ddataset)
    for model in session.model_options:
        session.add_model(bmds.constants.M_Weibull)
        session.add_model(bmds.constants.M_Weibull, overrides={'restrict_power': 0})

    session.execute()
    weibull1 = session.models[0]
    weibull2 = session.models[1]

    for m in session.models:
        assert m.output_created is True

    assert 'Power parameter is restricted as power >= 1' in weibull1.outfile
    assert 'Power parameter is not restricted' in weibull2.outfile


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

    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=bad_ddataset)
    session.add_default_models()
    session.execute()
    session.recommend()
    assert session.recommended_model_index is None
