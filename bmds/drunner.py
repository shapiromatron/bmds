import os

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

    def execute(self):

        for obj in self.inputs:

            # get executable path
            exe = session.BMDS.get_model(obj['bmds_version'], obj['model_name']).get_exe_path()

            # write dfile
            dfile = self.tempfiles.get_tempfile(prefix='bmds-dfile-', suffix='.(d)')
            with open(dfile, 'w') as f:
                f.write(obj['dfile'])

            output = {
                'output_created': False,
                'outfile': None,
            }
            try:
                utils.RunProcess([exe, dfile], timeout=20).call()
                outfile = self.get_outfile(dfile, obj['model_name'])
                oo2 = outfile.replace('.out', '.002')
                if os.path.exists(outfile):
                    self.tempfiles.append(outfile)
                    output['output_created'] = True
                    with open(outfile, 'r') as f:
                        output['outfile'] = f.read()
                if os.path.exists(oo2):
                    self.tempfiles.append(oo2)
            except Exception as e:
                raise e
            finally:
                self.tempfiles.cleanup()

            self.outputs.append(output)

        return self.outputs
