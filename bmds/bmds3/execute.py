import ctypes
import math
from pathlib import Path
import platform

from . import types


def get_dll_absolute_path() -> str:
    path = Path(__file__).absolute().parents[0]
    bits = platform.architecture()[0]
    if "64" in bits:
        return str(path / "bmds_models_x64.dll")
    elif "32" in bits:
        return str(path / "bmds_models.dll")
    else:
        raise OSError(f"Unknown arhictecture: {bits}")


def create_dichotomous_bmd_analysis(num_dg: int) -> types.BMD_ANAL:

    _dGoF_t = types.dGoF_t()
    _dGoF_t.pzRow = (types.GoFRow_t * num_dg)()

    analysis = types.BMD_ANAL()
    analysis.PARMS = (ctypes.c_double * types.MY_MAX_PARMS)()
    analysis.boundedParms = (ctypes.c_bool * types.MY_MAX_PARMS)()
    analysis.aCDF = (ctypes.c_double * types.CDF_TABLE_SIZE)()
    analysis.deviance = (types.DichotomousDeviance_t * num_dg)()
    analysis.gof = ctypes.pointer(_dGoF_t)
    analysis.nCDF = types.CDF_TABLE_SIZE

    return analysis


def build_dichotomous_dataset(doses, ns, incidences):
    # Returns an array of ctypes.ARRAY[types.BMDSInputData_t]
    datasets = [
        types.BMDSInputData_t(dose, n, incidence, 0.0)
        for dose, n, incidence in zip(doses, ns, incidences)
    ]
    return (types.BMDSInputData_t * len(datasets))(*datasets)


def bmds3_test():
    dll = ctypes.cdll.LoadLibrary(get_dll_absolute_path())
    dfunc = dll.run_dmodel2

    _DModelID_t = (ctypes.c_int * 1)(types.DModelID_t.eLogistic.value)
    _p_inputType = (ctypes.c_int * 1)(types.DModelID_t.eLogistic.value)

    # one row for each dose-group
    dataset = build_dichotomous_dataset([0, 20, 50, 100], [50, 50, 50, 50], [0, 4, 11, 13])
    results = create_dichotomous_bmd_analysis(num_dg=len(dataset))

    # logistic has two parameters, this is the prior for each of the params
    priors = (types.PRIOR * 2)(types.PRIOR(0, -2, 1, -18, 18), types.PRIOR(0, 0.1, 1, 1, 100))

    _BMDS_D_Opts1_t = types.BMDS_D_Opts1_t(0.1, 0.05, -9999)
    _BMDS_D_Opts2_t = types.BMDS_D_Opts2_t(1, 0)
    n = ctypes.c_int(len(dataset))

    dfunc(
        _DModelID_t,
        ctypes.pointer(results),
        _p_inputType,
        dataset,
        priors,
        ctypes.pointer(_BMDS_D_Opts1_t),
        ctypes.pointer(_BMDS_D_Opts2_t),
        ctypes.pointer(n),
    )

    print(results.BMDL, results.BMD, results.BMDU)
