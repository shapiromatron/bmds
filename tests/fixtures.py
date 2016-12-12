import pytest

import bmds
import numpy as np


@pytest.fixture
def cdataset():
    return bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[111, 142, 143, 93, 42],
        means=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159])


@pytest.fixture
def ddataset():
    return bmds.DichotomousDataset(
        doses=[0, 1.96, 5.69, 29.75],
        ns=[75, 49, 50, 49],
        incidences=[5, 1, 3, 14])


@pytest.fixture
def anova_dataset():
    variances = [0.884408974, 0.975597151, 0.301068371, 0.879069846, 0.220161323, 0.841362172, 0.571939331]
    stdevs = np.power(np.array(variances), 0.5)
    return bmds.ContinuousDataset(
        doses=[1, 2, 3, 4, 5, 6, 7],
        ns=[8, 6, 6, 6, 6, 6, 6],
        means=[9.9264, 10.18886667, 10.17755, 10.35711667, 10.02756667, 11.4933, 10.85275],
        stdevs=stdevs,
    )
