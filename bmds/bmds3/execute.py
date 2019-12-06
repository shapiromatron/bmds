import ctypes
from pathlib import Path
import platform

from . import types


def get_dll_absolute_path() -> str:
    path = Path(__file__).absolute().parents[0]
    bits = platform.architecture()[0]
    if '64' in bits:
        return str(path / 'bmds_models_x64.dll')
    elif '32' in bits:
        return str(path / 'bmds_models.dll')
    else:
        raise OSError(f"Unknown arhictecture: {bits}")


def bmds3_test():
    dll = ctypes.cdll.LoadLibrary(get_dll_absolute_path())
    dfunc = dll.run_dmodel2

    _DModelID_t = (ctypes.c_int * 1)(3)
    _p_inputType = (ctypes.c_int * 1)(3)
    # one row for each dose-group
    _BMDSInputData_t = (types.BMDSInputData_t * 4)(
        types.BMDSInputData_t( 0.0,  0, 50, 0),
        types.BMDSInputData_t( 2.0,  4, 50, 0),
        types.BMDSInputData_t( 5.0, 11, 50, 0),
        types.BMDSInputData_t(30.0, 13, 50, 0)
    )
    _BMD_ANAL = types.create_bmd_analysis(num_dg=4)
    # logistic has two parameters, this is the prior for each of the params
    _PRIOR = (types.PRIOR*2)(
        types.PRIOR(0, -2, 1, -18, 18),
        types.PRIOR(0, 0.1, 1, 1, 100)
    )
    _BMDS_D_Opts1_t = types.BMDS_D_Opts1_t(0.1, 0.05, -9999)
    _BMDS_D_Opts2_t = types.BMDS_D_Opts2_t(1, 0)
    _p_n = (ctypes.c_int * 1)(1)

    try:
        dfunc(
            ctypes.pointer(_DModelID_t),
            ctypes.pointer(_BMD_ANAL),
            ctypes.pointer(_p_inputType),
            _BMDSInputData_t,
            ctypes.pointer(_PRIOR),
            ctypes.pointer(_BMDS_D_Opts1_t),
            ctypes.pointer(_BMDS_D_Opts2_t),
            ctypes.pointer(_p_n)
        )
    except OSError:
        print('This is as far as I got....')
