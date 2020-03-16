from pathlib import Path

import pytest

from bmds.datasets import DichotomousDataset, ContinuousDataset
# from bmds.bmds3.models import dichotomous, continuous


@pytest.mark.skip(reason="todo - add back")
def test_bmds3_dichotomous():
    ds = DichotomousDataset(
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
        result = model.execute_dll(ds)
        print(result)


def _print_debug(name: str):
    print("-------------------------------------------------------")
    print(
        "\n".join(
            Path(rf"~\dev\bmds\{name}")
            .resolve()
            .read_text()
            .split("\n")[:30]
        )
    )


@pytest.mark.skip(reason="todo - add back")
def test_bmds3_continuous():
    ds = ContinuousDataset(
        doses=[0, 25, 50, 100, 200],
        ns=[20, 20, 20, 20, 20],
        means=[10.0, 15.0, 20.0, 25.0, 30.0],
        stdevs=[1, 2, 3, 4, 5],
    )
    models = [
        continuous.ExponentialM2(),
        continuous.ExponentialM3(),
        # continuous.ExponentialM4(),
        # continuous.ExponentialM5(),
        # continuous.Linear(),
        # continuous.Polynomial(degree=2),
        # continuous.Polynomial(degree=3),
        continuous.Power(),
        # continuous.Hill(),
    ]
    for model in models:
        model.execute_dll(ds)
    _print_debug("run_cmodel2.Power.log")
