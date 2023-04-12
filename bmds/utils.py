import ctypes
import os
import sys
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import tabulate

from .version import __version__

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


package_root = Path(__file__).absolute().parent


def multi_lstrip(txt: str) -> str:
    """Left-strip all lines in a multiline string."""
    return "\n".join(line.lstrip() for line in txt.splitlines()).strip()


def pretty_table(data, headers):
    return tabulate.tabulate(data, headers=headers, tablefmt="fancy_outline")


def ff(value) -> str:
    """Float formatter for floats and float-like values"""
    if isinstance(value, str):
        return value
    elif abs(value) > 1e6:
        return f"{value:.1E}"
    elif value > 0 and value < 0.001:
        return "<0.001"
    elif np.isclose(value, int(value)):
        return str(int(value))
    else:
        return f"{value:.3f}".rstrip("0")


def str_list(items: Iterable) -> str:
    return ",".join([str(item) for item in items])


def citation(dll_version: str | None = None) -> dict:
    """
    Return a citation for the software.
    """
    executed = datetime.now().strftime("%B %d, %Y")
    bmds_version = __version__
    url = "https://pypi.org/project/bmds/"
    if not dll_version:
        # assume we're using the latest version
        dll_version = get_latest_dll_version()
    return dict(
        paper=(
            "Pham LL, Watford S, Friedman KP, Wignall J, Shapiro AJ. Python BMDS: A Python "
            "interface library and web application for the canonical EPA dose-response modeling "
            "software. Reprod Toxicol. 2019;90:102-108. doi:10.1016/j.reprotox.2019.07.013."
        ),
        software=(
            f"Python BMDS. (Version {bmds_version}; Model Library Version {dll_version}) "
            f"[Python package]. Available from {url}. Executed on {executed}."
        ),
    )


def get_latest_dll_version() -> str:
    from .session import BMDS

    return BMDS.latest_version().dll_version()
