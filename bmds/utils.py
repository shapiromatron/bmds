import ctypes
import logging
from subprocess import Popen
import os
import sys
import tempfile
import threading


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

    def get_tempfile(self, prefix='', suffix='.txt'):
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


class RunProcess(threading.Thread):
    """
    Run process and terminate a process given a specified time interval.

    Adapted from: http://stackoverflow/questions/4158502/

    Two input arguments:
        - cmd, the command to execute in subprocess, as a list
        - timeout, the maximum length of time to execute, in seconds

    Example call:
        - RunProcess(["./foo", "arg1"]).call()
    """

    def __init__(self, cmd, timeout=60):
        threading.Thread.__init__(self)
        self.cmd = cmd
        self.timeout = timeout

    def run(self):
        self.p = Popen(self.cmd)
        self.p.wait()

    def handle_failure(self):
        pass

    def call(self):
        self.start()
        self.join(self.timeout)
        if self.is_alive():
            self.p.terminate()
            self.handle_failure()
            self.join()


class RunBMDS(RunProcess):
    """
    Custom thread class for running BMDS models. Logs failures when BMDS
    models are killed, including printing the (d) file.
    """
    
    def handle_failure(self):
        exe = self.cmd[0]
        with open(self.cmd[1], 'r') as f:
            dfile = f.read()
        logger.warning('BMDS model killed: {}\n{}'.format(exe, dfile))
