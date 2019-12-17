import ctypes
import logging
import os
from pathlib import Path
import platform
import sys
import tempfile

from typing import Callable


logger = logging.getLogger(__name__)


# http://stackoverflow.com/questions/24130623/
# Don't display the Windows GPF dialog if the invoked program dies.
# Required for Weibull model with some datasets with negative slope
SUBPROCESS_FLAGS = 0
if sys.platform.startswith("win"):
    SEM_NOGPFAULTERRORBOX = 0x0002  # From MSDN
    ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class TempFileList(list):
    # Maintains a list of temporary files and cleans up after itself

    def get_tempfile(self, prefix="", suffix=".txt"):
        fd, fn = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        os.close(fd)
        self.append(fn)
        return fn

    def cleanup(self):
        for fn in iter(self):
            try:
                os.remove(fn)
            except OSError:
                pass

    def __del__(self):
        self.cleanup()


def get_dll_func(bmds_version: str, base_name: str, func_name: str) -> Callable:
    """
    Return a callable function from a dll. The filename will be OS and environment specific.

    Args:
        bmds_version (str): The bmds version, eg., `BMDS312`
        base_name (str): The base-name for the file eg., `bmds_models`
        func_name (str): The callable function from the dll, eg., `run_cmodel`

    Raises:
        EnvironmentError: System could not be determined
        FileNotFoundError: The dll file could not be found

    Returns:
        Callable: the callable function from the dll
    """
    filename = base_name

    bits = platform.architecture()[0]
    if "64" in bits:
        filename += "_x64"
    elif "32" in bits:
        pass
    else:
        raise EnvironmentError(f"Unknown architecture: {bits}")

    os_ = platform.system()
    if os_ == "Windows":
        filename += ".dll"
    elif os_ in ("Darwin", "Linux"):
        filename += ".so"
    else:
        raise EnvironmentError(f"Unknown OS: {os_}")

    path = Path(__file__).absolute().parents[0] / "bin" / bmds_version / filename
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    dll = ctypes.cdll.LoadLibrary(str(path))
    return getattr(dll, func_name)
