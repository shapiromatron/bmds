from ..datasets import DichotomousDataset, ContinuousDataset
from .models import dichotomous, continuous


def bmds3_test():
    ds = DichotomousDataset(
        doses=[0, 20, 50, 100], ns=[50, 50, 50, 50], incidences=[0, 4, 11, 13]
    )
    models = [
        dichotomous.Logistic(),
        # dichotomous.Multistage(degree=1),
        # dichotomous.Multistage(degree=2),
        # dichotomous.Multistage(degree=3),
        # dichotomous.LogLogistic(),
        # dichotomous.Probit(),
        # dichotomous.LogProbit(),
        # dichotomous.Gamma(),
        # dichotomous.QuantalLinear(),
        # dichotomous.Weibull(),
        # dichotomous.DichotomousHill(),
    ]
    for model in models:
        print(model.execute_dll(ds))

    ds = ContinuousDataset(
        doses=[0, 25, 50, 100, 200],
        ns=[20, 20, 19, 20, 20],
        means=[10, 11, 13, 17, 22],
        stdevs=[2, 2, 2, 3, 4],
    )
    models = [
        # continuous.ExponentialM2(),
        # continuous.ExponentialM3(),
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
