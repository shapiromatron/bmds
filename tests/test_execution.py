import os
import inspect

import pytest

import bmds


@pytest.fixture
def dataset():
    return bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[111, 142, 143, 93, 42],
        responses=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159])


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


def test_model_execute(dataset):
    model = bmds.models.Power_218(dataset)
    model.execute()
    assert model.output_created is True


def test_session_execute(dataset):
    session = bmds.BMDS_v2601(bmds.constants.CONTINUOUS, dataset=dataset)
    for model in session.model_options:
        session.add_model(bmds.constants.M_Power)
        session.add_model(bmds.constants.M_Polynomial)
    session.execute()
    assert session._models[0].output_created is True
    assert session._models[1].output_created is True

    # check that polynomial restriction is being used
    assert 'The polynomial coefficients are restricted to be positive' in session._models[1].outfile
