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
    """
    To run:
    1) rerun the install script `pip install -e .`
    2) Try using the new CLI `bmds3_test`

    I'd probably pull all of this out into a jupyter notebook to play around with until we figure
    out what sticks...

    You can also use `import pdb; pdb.set_trace()` which can be pretty handy.
    """
    print('starting test')
    dll = ctypes.cdll.LoadLibrary(get_dll_absolute_path())
    dfunc = dll.run_dmodel2

    _DModelID_t = (ctypes.c_int * 1)()
    _DModelID_t[0] = 3

    _BMD_ANAL = types.BMD_ANAL()

    _p_inputType = (ctypes.c_int * 1)()
    _p_inputType[0] = 3

    _BMDSInputData_t = 3

    _PRIOR = types.PRIOR()
    data = [0, 0, 0, 0, 0]
    _PRIOR.data = (ctypes.c_double * len(data))(*data)

    # TODO: RESUME HERE
    try:
        dfunc(
            ctypes.pointer(ctypes.c_int(3)),
            ctypes.pointer(_BMD_ANAL),
            ctypes.pointer(types.BMDSInputType_t.eDich_3),      # todo change to ctypes array?
            # ....
        )
    except TypeError:
        print('This is as far as I got....')
