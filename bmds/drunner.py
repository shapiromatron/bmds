import os

from . import bmds, utils

from .models.base import TempFileMaker


class BatchDfileRunner(TempFileMaker):
    """
    Batch-execute a list of pre-created d-files.

    Used to execute BMDS on a remote Windows server, if for example you're
    running a library on a non-Windows OS.
    """

    def __init__(self, inputs):
        super(TempFileMaker, self).__init__()
        self.inputs = inputs
        self.outputs = None
        self.execute()

    def execute(self):

        for obj in self.inputs:

            # get executable path
            exe = bmds.get_model(obj['bmds_version'], obj['model_app_name']).get_exe_path()

            # write dfile
            dfile = self.get_tempfile(prefix='bmds-dfile-', suffix='.(d)')
            with open(dfile, 'w') as f:
                f.write(obj['dfile'])

            output = {
                'output_created': False,
                'outfile': None,
            }
            try:
                utils.RunProcess([exe, dfile], timeout=20).call()
                outfile = dfile.replace('.(d)', '.out')
                if os.path.exists(outfile):
                    output['output_created'] = True
                    self.add_tempfile(outfile)
                    with open(outfile, 'r') as f:
                        output['outfile'] = f.read()
                o2 = dfile.replace('.(d)', '.002')
                if os.path.exists(o2):
                    self.add_tempfile(o2)
            except Exception as e:
                raise e
            finally:
                self.cleanup()

            self.outputs.append(output)

        return self.outputs
