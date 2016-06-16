import os
import tempfile

from .. import constants
from ..parser import OutputParser
from ..utils import RunProcess


BMR_CROSSWALK = {
    constants.DICHOTOMOUS: {
        'Extra': 0,
        'Added': 1
    },
    constants.DICHOTOMOUS_CANCER: {
        'Extra': 0
    },
    constants.CONTINUOUS: {
        'Abs. Dev.': 0,
        'Std. Dev.': 1,
        'Rel. Dev.': 2,
        'Point': 3,
        'Extra': 4
    }
}


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bin'))


class BMDModel(object):
    """
    Has the following required static-class methods:

        - model_name = ''      # string model name
        - dtype = ''           # data type - 'D','C', etc.
        - exe = ''             # BMD executable (without extension)
        - exe_plot             # wgnuplot input-file executable (w/o extension)
        - version = 0          # version number
        - date = ''            # version date
        - defaults = {}        # default options setup
        - possible_bmr = ()    # possible BMRs which can be used w/ model

    And at-least these instance methods:

        - self.override = {}        # overridden values from default
        - self.override_txt = ['']  # text string(s) for overridden values
        - self.values = {}          # full values for object

    Default key fields:
      - c = category
      - t = type
      - f = fixed
      - n = name

    """

    def __init__(self, dataset):

        self.dataset = dataset

        self.override = {}
        self.override_txt = ['']

        # set default values
        self.values = {}
        for k, v in self.defaults.iteritems():
            self.values[k] = self._get_option_value(k)

        self.tempfns = []
        self.output_created = False

    def execute(self):
        try:
            exe = self.get_exe_path()
            dfile = self.write_dfile()

            RunProcess([exe, dfile], timeout=20).call()

            outfile = dfile.replace('.(d)', '.out')
            if os.path.exists(outfile):
                self.output_created = True
                self.tempfns.append(outfile)
                self.parse_results(outfile)
            
            o2 = dfile.replace('.(d)', '.002')
            if os.path.exists(o2):
                self.tempfns.append(o2)

        except Exception as e:
            raise e
        finally:
            self.cleanup()

    def get_tempfile(self, prefix='bmds-', suffix='.txt'):
        fd, fn = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        os.close(fd)
        self.tempfns.append(fn)
        return fn

    def cleanup(self):
        for fn in self.tempfns:
            os.remove(fn)
                
    @classmethod
    def get_exe_path(cls):
        return os.path.abspath(os.path.join(
            ROOT,
            cls.bmds_version_dir,
            cls.exe + '.exe'))

    def parse_results(self, fn):
        with open(fn, 'r') as f:
            text = f.read()
        parser = OutputParser(text, self.dtype, self.model_name)
        self.output_text = text
        self.output = parser.output

    def as_dfile(self):
        raise NotImplementedError('Abstract method requires implementation')

    def write_dfile(self):
        f_in = self.get_tempfile(suffix='.(d)')
        with open(f_in, 'w') as f:
            f.write(self.as_dfile())
        return f_in

    def _get_option_value(self, key):
        """
        Get option value(s), or use default value if no override value.
        Two output values for 'p' type values (parameters), else one.
        Returns a tuple of two values.
        """
        if key in self.override:
            val = self.override[key]
        else:
            val = self.defaults[key]['d']

        if self.defaults[key]['t'] == 'p':  # parameter (two values)
            return val.split('|')
        else:
            return val, False

    def _dfile_print_header_rows(self):
        return '{}\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out'.format(self.model_name)  # noqa

    def _dfile_print_parameters(self, *params):
        # Print parameters in the specified order. Expects a tuple of parameter
        # names, in the proper order.
        if ((self.dtype == constants.CONTINUOUS) and
                (self.values['constant_variance'][0] == 1)):
            self.values['rho'] = ('s', 0)  # for specified to equal 0
        specifieds = []
        initials = []
        init = '0'  # 1 if initialized, 0 otherwise
        for param in params:
            t, v = self.values[param]
            # now add values
            if t == 'd':
                specifieds.append(-9999)
                initials.append(-9999)
            elif t == 's':
                specifieds.append(v)
                initials.append(-9999)
            elif t == 'i':
                init = '1'
                specifieds.append(-9999)
                initials.append(v)

        return '\n'.join([
            ' '.join([str(i) for i in specifieds]),
            init,
            ' '.join([str(i) for i in initials])
        ])

    def _dfile_print_options(self, *params):
        # Return space-separated list of values for dfile
        return ' '.join([str(self.values[param][0]) for param in params])
