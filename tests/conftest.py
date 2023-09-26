from pathlib import Path

import numpy as np
import pytest

import bmds


@pytest.fixture(scope="session")
def data_path():
    return Path(__file__).parent.absolute() / "data"


@pytest.fixture(scope="session")
def rewrite_data_files():
    """
    If you're making changes to datasets and it's expected that previously saved data will need to
    be written, then you can set this flag to True and then all saved data will be rewritten.

    Please review changes to ensure they're expected after modifying this flag.

    A test exists in CI to ensure that this flag is set to False on commit.
    """
    return False


@pytest.fixture
def cdataset():
    return bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[111, 142, 143, 93, 42],
        means=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159],
    )


@pytest.fixture
def cdataset2():
    return bmds.ContinuousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[100, 100, 100, 100, 100],
        means=[10, 20, 30, 40, 50],
        stdevs=[3, 4, 5, 6, 7],
    )


@pytest.fixture
def negative_cdataset():
    return bmds.ContinuousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[100, 100, 100, 100, 100],
        means=[1, -5, -10, -20, -30],
        stdevs=[3, 4, 5, 6, 7],
    )


@pytest.fixture
def cidataset():
    # fmt: off
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
        ],
    )
    # fmt: on


@pytest.fixture
def ddataset():
    return bmds.DichotomousDataset(
        doses=[0, 1.96, 5.69, 29.75], ns=[75, 49, 50, 49], incidences=[5, 1, 3, 14]
    )


@pytest.fixture
def ddataset2():
    return bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )


@pytest.fixture
def ddataset3():
    # float based count data based on possible data transformations
    return bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[10.1, 10.2, 10.3, 10.4, 10.5],
        incidences=[0.0, 2.5, 3.5, 7.5, 9.5],
    )


@pytest.fixture
def anova_dataset():
    variances = [
        0.884408974,
        0.975597151,
        0.301068371,
        0.879069846,
        0.220161323,
        0.841362172,
        0.571939331,
    ]
    stdevs = np.power(np.array(variances), 0.5)
    return bmds.ContinuousDataset(
        doses=[1, 2, 3, 4, 5, 6, 7],
        ns=[8, 6, 6, 6, 6, 6, 6],
        means=[9.9264, 10.18886667, 10.17755, 10.35711667, 10.02756667, 11.4933, 10.85275],
        stdevs=stdevs.tolist(),
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
        doses=[0, 0.2, 1.5], ns=[6, 6, 6], means=[0.0, 0.0, 0.0], stdevs=[0.0, 0.0, 0.0]
    )


@pytest.fixture
def bad_ddataset():
    return bmds.DichotomousDataset(
        doses=[0, 5.0, 50.0, 150.0], ns=[10, 10, 10, 10], incidences=[0, 0, 0, 0]
    )


@pytest.fixture
def reduced_cdataset():
    # dataset results in reduced model form fits
    # Exponential M3 -> M2
    # Exponential M5 -> M4
    # Multi/Power -> Linear
    return bmds.ContinuousDataset(
        doses=[0.0, 1340.0, 2820.0, 5600.0, 11125.0, 23000.0],
        ns=[10, 10, 10, 10, 10, 8],
        means=[29.3, 28.4, 27.2, 26.0, 25.8, 24.5],
        stdevs=[2.53, 1.9, 2.53, 2.53, 2.21, 1.58],
    )


@pytest.fixture
def ddataset_requires_dose_drop():
    # a dataset which requires doses to be dropped before a model can be recommended
    return bmds.DichotomousDataset(
        doses=[0.0, 10.0, 20.0, 40.0, 80.0, 120.0],
        ns=[10, 10, 10, 10, 10, 10],
        incidences=[0, 0, 3, 4, 6, 0],
    )


@pytest.fixture
def nd_dataset():
    # fmt: off
    return bmds.NestedDichotomousDataset(
        id=123,
        doses=[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            25, 25, 25, 25, 25, 25, 25, 25, 25,
            50, 50, 50, 50, 50, 50, 50, 50, 50,
        ],
        litter_ns=[
            16, 9, 15, 14, 13, 9, 10, 14, 10, 11, 14,
            9, 14, 9, 13, 12, 10, 10, 11, 14,
            11, 11, 14, 11, 10, 11, 10, 15, 7,
        ],
        incidences=[
            1, 1, 2, 3, 3, 0, 2, 2, 1, 2, 4,
            5, 6, 2, 6, 3, 1, 2, 4, 3,
            4, 5, 5, 4, 5, 4, 5, 6, 2,
        ],
        litter_covariates=[
            16, 9, 15, 14, 13, 9, 10, 14, 10, 11, 14,
            9, 14, 9, 13, 12, 10, 10, 11, 14,
            11, 11, 14, 11, 10, 11, 10, 15, 7,
        ]
    )
    # fmt: on


@pytest.fixture
def nd_dataset4():
    # fmt: off
    return bmds.NestedDichotomousDataset(
        id=123,
        doses=[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            25, 25, 25, 25, 25, 25, 25,
            50, 50, 50, 50, 50, 50,
            100, 100, 100, 100, 100,
        ],
        litter_ns=[
            16, 9, 15, 14, 13, 9, 10, 14, 10, 11, 14,
            9, 14, 9, 13, 12, 10, 10,
            11, 14, 11, 11, 14, 11,
            10, 11, 10, 15, 7,
        ],
        incidences=[
            1, 1, 2, 3, 3, 0, 2, 2, 1, 2, 4,
            5, 6, 2, 6, 3, 1, 2,
            4, 3, 4, 5, 5, 4,
            5, 4, 5, 6, 2,
        ],
        litter_covariates=[
            16, 9, 15, 14, 13, 9, 10, 14, 10, 11, 14,
            9, 14, 9, 13, 12, 10, 10,
            11, 14, 11, 11, 14, 11,
            10, 11, 10, 15, 7,
        ]
    )
    # fmt: on
