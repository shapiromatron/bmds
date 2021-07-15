import ctypes
import os
import sys
import tempfile
from pathlib import Path

import tabulate

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
    return tabulate.tabulate(data, headers=headers, tablefmt="fancy_grid")
