import asyncio
import os
from simple_settings import settings
import sys

from . import utils, session


class BatchDfileRunner(object):
    """
    Batch-execute a list of pre-created d-files.

    Used to execute BMDS on a remote Windows server, if for example you're
    running a library on a non-Windows OS.
    """

    def __init__(self, inputs):
        self.tempfiles = utils.TempFileList()
        self.inputs = inputs
        self.outputs = []
        self.execute()

    def get_outfile(self, dfile, model_name):
        outfile = dfile.replace('.(d)', '.out')
        oo2 = outfile.replace('.out', '.002')

        # if not exponential, exit early
        if 'exponential' not in model_name.lower():
            return outfile

        # side-effect- cleanup other files created by exponential
        if os.path.exists(outfile):
            self.tempfiles.append(outfile)
        if os.path.exists(oo2):
            self.tempfiles.append(oo2)

        # get exponential model prefix
        prefix = model_name.split('-')[1]
        path, fn = os.path.split(outfile)
        outfile = os.path.join(path, prefix + fn)
        return outfile

    async def execute_job(self, obj):
        """
        Execute the BMDS model and parse outputs if successful.
        """

        # get executable path
        exe = session.BMDS\
            .get_model(obj['bmds_version'], obj['model_name'])\
            .get_exe_path()

        # write dfile
        dfile = self.tempfiles.get_tempfile(prefix='bmds-dfile-', suffix='.(d)')
        with open(dfile, 'w') as f:
            f.write(obj['dfile'])

        output = {
            'output_created': False,
            'execution_halted': False,
            'outfile': None,
        }

        proc = await asyncio.create_subprocess_exec(
                exe, dfile,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=settings.BMDS_MODEL_TIMEOUT_SECONDS)

        except asyncio.TimeoutError:
            proc.kill()
            output['execution_halted'] = True
            stdout, stderr = await proc.communicate()

        outfile = self.get_outfile(dfile, obj['model_name'])
        oo2 = outfile.replace('.out', '.002')
        if os.path.exists(outfile):
            self.tempfiles.append(outfile)
            output['output_created'] = True
            with open(outfile, 'r') as f:
                output['outfile'] = f.read()
        if os.path.exists(oo2):
            self.tempfiles.append(oo2)

        output['stdout'] = stdout.decode().strip()
        output['stderr'] = stderr.decode().strip()

        self.tempfiles.cleanup()
        self.outputs.append(output)


    async def execute_jobs(self, objects):
        await asyncio.wait([
            self.execute_job(obj) for obj in objects
        ])

    def execute(self):
        if sys.platform == 'win32':
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)
        else:
            loop = asyncio.get_event_loop()
        loop.run_until_complete(self.execute_jobs(self.inputs))
        return self.outputs
