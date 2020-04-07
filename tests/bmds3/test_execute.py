import platform

import pytest

import bmds

is_windows = platform.system() == "Windows"

if is_windows:
    from bmds.bmds3.models import dichotomous, continuous


@pytest.mark.skipif(not is_windows, reason="dlls only exist for Windows")
def test_bmds3_dichotomous():
    ds = bmds.datasets.DichotomousDataset(
        doses=[0, 20, 50, 100], ns=[50, 50, 50, 50], incidences=[0, 4, 11, 13]
    )
    models = [
        dichotomous.Logistic(),
        dichotomous.LogLogistic(),
        dichotomous.Probit(),
        dichotomous.LogProbit(),
        dichotomous.Gamma(),
        dichotomous.QuantalLinear(),
        dichotomous.Weibull(),
        dichotomous.DichotomousHill(),
        dichotomous.Multistage(degree=2),
        dichotomous.Multistage(degree=3),
    ]
    for model in models:
        result = model.execute(ds)
        print(result)


@pytest.mark.skipif(not is_windows, reason="dlls only exist for Windows")
def test_bmds3_continuous():
    ds = bmds.datasets.ContinuousDataset(
        doses=[0, 25, 50, 100, 200],
        ns=[20, 20, 20, 20, 20],
        means=[10.0, 15.0, 20.0, 25.0, 30.0],
        stdevs=[1, 2, 3, 4, 5],
    )
    models = [
        continuous.ExponentialM2(),
        continuous.ExponentialM3(),
        continuous.ExponentialM4(),
        continuous.ExponentialM5(),
        continuous.Linear(),
        continuous.Polynomial(degree=2),
        continuous.Polynomial(degree=3),
        continuous.Power(),
        continuous.Hill(),
    ]
    for model in models:
        result = model.execute(ds)
        print(result)
