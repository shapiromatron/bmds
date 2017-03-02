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
def cidataset():
    return bmds.ContinuousIndividualDataset(
        doses=[
            0, 0, 0, 0, 0, 0, 0, 0,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            1, 1, 1, 1, 1, 1,
            10, 10, 10, 10, 10, 10,
            100, 100, 100, 100, 100, 100,
            300, 300, 300, 300, 300, 300,
            500, 500, 500, 500, 500, 500,
        ],
        responses=[
            8.1079, 9.3063, 9.7431, 9.7814, 10.0517, 10.6132, 10.7509, 11.0567,
            9.1556, 9.6821, 9.8256, 10.2095, 10.2222, 12.0382,
            9.5661, 9.7059, 9.9905, 10.2716, 10.471, 11.0602,
            8.8514, 10.0107, 10.0854, 10.5683, 11.1394, 11.4875,
            9.5427, 9.7211, 9.8267, 10.0231, 10.1833, 10.8685,
            10.368, 10.5176, 11.3168, 12.002, 12.1186, 12.6368,
            9.9572, 10.1347, 10.7743, 11.0571, 11.1564, 12.0368,
        ])


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


@pytest.fixture
def bad_anova_dataset():
    return bmds.ContinuousDataset(
        doses=[0.0, 0.2, 1.5, 10.0],
        means=[0.0, 0.0, 0.0, 0.67],
        ns=[6, 6, 6, 4],
        stdevs=[0.0, 0.0, 0.0, 0.67],
    )


@pytest.fixture
def bad_cdataset():
    return bmds.ContinuousDataset(
        doses=[0, 0.2, 1.5],
        ns=[6, 6, 6],
        means=[0.0, 0.0, 0.0],
        stdevs=[0.0, 0.0, 0.0]
    )


@pytest.fixture
def bad_ddataset():
    return bmds.DichotomousDataset(
        doses=[0, 5.0, 50.0, 150.0],
        ns=[10, 10, 10, 10],
        incidences=[0, 0, 0, 0]
    )
