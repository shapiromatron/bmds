import os
import subprocess

from simple_settings import settings

from .. import session, utils
from .models.base import RunStatus


class BatchDfileRunner:
    """
    Batch-execute a list of pre-created d-files.

    Used to execute BMDS on a remote Windows server, if for example you're
    running a library on a non-Windows OS.
    """

    def __init__(self, inputs):
        self.tempfiles = utils.TempFileList()
        self.inputs = inputs

    def get_outfile(self, dfile, model_name):
        outfile = dfile.replace(".(d)", ".out")
        oo2 = outfile.replace(".out", ".002")

        # if not exponential, exit early
        if "exponential" not in model_name.lower():
            return outfile

        # side-effect- cleanup other files created by exponential
        if os.path.exists(outfile):
            self.tempfiles.append(outfile)
        if os.path.exists(oo2):
            self.tempfiles.append(oo2)

        # get exponential model prefix
        prefix = model_name.split("-")[1]
        path, fn = os.path.split(outfile)
        outfile = os.path.join(path, prefix + fn)
        return outfile

    def execute_job(self, obj):
        """
        Execute the BMDS model and parse outputs if successful.
        """

        # get executable path
        exe = session.BMDS.get_model(obj["bmds_version"], obj["model_name"]).get_exe_path()

        # write dfile
        dfile = self.tempfiles.get_tempfile(prefix="bmds-dfile-", suffix=".(d)")
        with open(dfile, "w") as f:
            f.write(obj["dfile"])

        outfile = self.get_outfile(dfile, obj["model_name"])
        oo2 = outfile.replace(".out", ".002")

        proc = subprocess.Popen([exe, dfile], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        output = None
        stdout = ""
        stderr = ""

        try:
            stdout, stderr = proc.communicate(timeout=settings.BMDS_MODEL_TIMEOUT_SECONDS)

            if os.path.exists(outfile):
                with open(outfile, "r") as f:
                    output = f.read()

            status = RunStatus.SUCCESS.value
            stdout = stdout.decode().strip()
            stderr = stderr.decode().strip()

        except subprocess.TimeoutExpired:
            proc.kill()
            status = RunStatus.FAILURE.value
            stdout, stderr = proc.communicate()

        finally:
            if os.path.exists(outfile):
                self.tempfiles.append(outfile)
            if os.path.exists(oo2):
                self.tempfiles.append(oo2)

        self.tempfiles.cleanup()

        return dict(status=status, output=output, stdout=stdout, stderr=stderr)

    def execute(self):
        return list(map(self.execute_job, self.inputs))
