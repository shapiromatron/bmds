import numpy as np

from bmds import bmdscore
from bmds.bmds3.constants import N_BMD_DIST
from bmds.bmds3.models.dichotomous import Multistage
from bmds.bmds3.types.dichotomous import DichotomousModelSettings
from bmds.datasets import DichotomousDataset


def get_multitumor_prior(degree):
    items = [[0, -17, 0, -18, 18]] + [[0, 0.1, 0, 0, 1e4]] * degree
    return [item for sublist in zip(*items, strict=True) for item in sublist]


def test_execution():
    # if degree is zero, loop for degree 2 to n-1 dose groups
    # if degree is nonzero, only use that value

    ds1 = DichotomousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[100, 100, 100, 100, 100],
        incidences=[0, 5, 30, 65, 90],
    )
    ds2 = DichotomousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[100, 100, 100, 100, 100],
        incidences=[5, 10, 33, 67, 93],
    )
    ds3 = DichotomousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[100, 100, 100, 100, 100],
        incidences=[1, 68, 78, 88, 98],
    )
    datasets = [ds1, ds2, ds3]
    degrees = [3, 0, 0]

    n_datasets = len(datasets)
    models = []
    results = []
    ns = []
    for i, dataset in enumerate(datasets):
        model_i = []
        result_i = []
        ns.append(dataset.num_dose_groups)
        degree_i = degrees[i]
        degrees_i = (
            range(degree_i, degree_i + 1) if degree_i > 0 else range(2, dataset.num_dose_groups)
        )
        for degree in degrees_i:
            # build inputs
            settings = DichotomousModelSettings(degree=degree)
            d = Multistage(dataset, settings=settings)
            inputs = d._build_inputs()
            analysis = inputs.to_cpp_analysis()
            analysis.degree = degree
            analysis.prior = get_multitumor_prior(degree)
            analysis.burnin = 0
            analysis.samples = 0
            model_i.append(analysis)

            # build outputs
            res = bmdscore.python_dichotomous_model_result()
            res.model = bmdscore.dich_model.d_multistage
            res.nparms = degree + 1
            res.dist_numE = N_BMD_DIST * 2
            res.gof = bmdscore.dichotomous_GOF()
            res.bmdsRes = bmdscore.BMDS_results()
            res.aod = bmdscore.dicho_AOD()
            result_i.append(res)

        models.append(model_i)
        results.append(result_i)

    analysis = bmdscore.python_multitumor_analysis()
    analysis.BMD_type = 1
    analysis.BMR = 0.1
    analysis.alpha = 0.05
    analysis.degree = degrees
    analysis.models = models
    analysis.n = ns
    analysis.ndatasets = n_datasets
    analysis.nmodels = [len(model_degree) for model_degree in models]
    # analysis.prior: list[list[float]] = ...
    analysis.prior_cols = 5

    result = bmdscore.python_multitumor_result()
    result.ndatasets = n_datasets
    result.nmodels = [len(model_degree) for model_degree in models]
    result.models = results

    bmdscore.pythonBMDSMultitumor(analysis, result)
    assert result.BMD == 6.005009
