import logging
from subprocess import Popen
import os
import tempfile
import threading


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

    def call(self):
        self.start()
        self.join(self.timeout)

        if self.is_alive():
            logging.warning("Process stopped; timeout: %s" %
                            ' '.join(self.cmd))
            self.p.terminate()
            self.join()
