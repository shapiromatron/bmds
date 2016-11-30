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


def test_model_execute(cdataset):
    model = bmds.models.Power_218(cdataset)
    model.execute()
    assert model.output_created is True


def test_session_execute(cdataset):
    session = bmds.BMDS_v2601(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_model(bmds.constants.M_Power)
    session.add_model(bmds.constants.M_Polynomial)
    session.execute()
    assert session._models[0].output_created is True
    assert session._models[1].output_created is True

    # check that polynomial restriction is being used
    assert 'The polynomial coefficients are restricted to be negative' in session._models[1].outfile


def test_parameter_overrides(cdataset):
    # assert to overrides are used
    session = bmds.BMDS_v2601(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_model(bmds.constants.M_Polynomial)
    session.add_model(bmds.constants.M_Polynomial,
                      overrides={'constant_variance': 1, 'degree_poly': 3})

    session.execute()
    model1 = session._models[0]
    model2 = session._models[1]

    assert model1.output_created is True and model2.output_created is True

    # check model-variance setting
    assert 'The variance is to be modeled' in model1.outfile
    assert 'rho' in model1.output['parameters']
    assert 'A constant variance model is fit' in model2.outfile
    assert 'rho' not in model2.output['parameters']

    # check degree_poly override setting
    assert 'beta_3' not in model1.output['parameters']
    assert model2.output['parameters']['beta_3']['estimate'] == 0.0


def test_continuous_restrictions(cdataset):
    session = bmds.BMDS_v2601(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_model(bmds.constants.M_Power)
    session.add_model(bmds.constants.M_Power, overrides={'restrict_power': 0})
    session.add_model(bmds.constants.M_Hill)
    session.add_model(bmds.constants.M_Hill, overrides={'restrict_n': 0})

    session.execute()
    power1 = session._models[0]
    power2 = session._models[1]
    hill1 = session._models[2]
    hill2 = session._models[3]

    for m in session._models:
        assert m.output_created is True

    assert 'The power is restricted to be greater than or equal to 1' in power1.outfile
    assert 'The power is not restricted' in power2.outfile
    assert 'Power parameter restricted to be greater than 1' in hill1.outfile
    assert 'Power parameter is not restricted' in hill2.outfile


def test_dichotomous_restrictions(ddataset):
    session = bmds.BMDS_v2601(bmds.constants.DICHOTOMOUS, dataset=ddataset)
    for model in session.model_options:
        session.add_model(bmds.constants.M_Weibull)
        session.add_model(bmds.constants.M_Weibull, overrides={'restrict_power': 0})

    session.execute()
    weibull1 = session._models[0]
    weibull2 = session._models[1]

    for m in session._models:
        assert m.output_created is True

    assert 'Power parameter is restricted as power >= 1' in weibull1.outfile
    assert 'Power parameter is not restricted' in weibull2.outfile
