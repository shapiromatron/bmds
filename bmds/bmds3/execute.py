import ctypes
from typing import Collection, Tuple

from . import types
from ..utils import get_dll_func
from ..datasets import DichotomousDataset
from .models import dichotomous


def prepare_continuous_data(
    doses: Collection[float],
    ns: Collection[int],
    means: Collection[float],
    stdevs: Collection[float]
) -> Tuple[ctypes.Array, types.BMD_ANAL]:
    """
    Given a dataset, create the required input/output data slots required for model execution.

    Args:
        doses (Collection[float]): [description]
        ns (Collection[int]): [description]
        means (Collection[float]): [description]
        stdevs (Collection[float]): [description]

    Returns:
        Tuple[ctypes.Array[types.BMDSInputData_t], types.BMD_ANAL]: [description]
    """
    num_dg = len(doses)
    datasets = (types.BMDSInputData_t * num_dg)(*[
        types.BMDSInputData_t(dose, n, mean, stdev)
        for dose, n, mean, stdev in zip(doses, ns, means, stdevs)
    ])

    analysis = types.BMD_C_ANAL()
    analysis.deviance = types.ContinuousDeviance_t(
        llRows=(types.LLRow_t * types.NUM_LIKELIHOODS_OF_INTEREST)(),
        testRows=(types.TestRow_t * types.NUM_TESTS_OF_INTEREST)()
    )
    analysis.PARMS = (ctypes.c_double * types.MY_MAX_PARMS)()
    analysis.gofRow = (types.cGoFRow_t * num_dg)()
    analysis.boundedParms = (ctypes.c_bool * types.MY_MAX_PARMS)()
    analysis.aCDF = (ctypes.c_double * types.CDF_TABLE_SIZE)()
    analysis.nCDF = types.CDF_TABLE_SIZE

    return datasets, analysis


def continuous_test():
    func = get_dll_func(bmds_version="BMDS312", base_name="cmodels", func_name="run_cmodel")

    model_id = (ctypes.c_int * 1)(types.CModelID_t.ePow.value)
    model_type = (ctypes.c_int * 1)(types.CModelID_t.ePow.value)

    # one row for each dose-group
    dataset, results = prepare_continuous_data(
        doses=[0, 25, 50, 100, 200],
        ns=[20, 20, 19, 20, 20],
        means=[6.0, 5.2, 2.4, 1.1, 0.75],
        stdevs=[1.2, 1.1, 0.81, 0.74, 0.66]
    )
    n = ctypes.c_int(len(dataset))

    # using default priors
    priors_ = [
        types.PRIOR(type=0, initialValue=1, stdDev=1, minValue=0, maxValue=1e8),
        types.PRIOR(type=0, initialValue=3.71e-03, stdDev=1, minValue=0, maxValue=1e8),
        types.PRIOR(type=0, initialValue=9.2965, stdDev=1, minValue=0, maxValue=1e8),
        types.PRIOR(type=0, initialValue=1.77258, stdDev=1, minValue=1, maxValue=1000),
        types.PRIOR(type=0, initialValue=1.93612, stdDev=1, minValue=-1000, maxValue=1000),
    ]
    priors = (types.PRIOR * len(priors_))(*priors_)

    options = types.BMDS_C_Options_t(
        bmr=0.1,
        alpha=0.05,
        background=-9999,
        tailProb=0.01,
        bmrType=types.BMRType_t.eRelativeDev.value,
        degree=0,
        adverseDirection=0,
        restriction=1,
        varType=types.VarType_t.eConstant.value,
        bLognormal=False,
        bUserParmInit=False,
    )

    func(
        model_id,
        ctypes.pointer(results),
        model_type,
        dataset,
        priors,
        ctypes.pointer(options),
        ctypes.pointer(n),
    )

    print(f'Continuous: BMDL: {results.BMDL:.3f} BMD: {results.BMD:.3f} BMDU: {results.BMDU:.3f}')


def bmds3_test():
    dataset = DichotomousDataset(
        doses=[0, 20, 50, 100],
        ns=[50, 50, 50, 50],
        incidences=[0, 4, 11, 13]
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
    ]
    for model in models:
        print(model.execute_dll(dataset))

    continuous_test()
