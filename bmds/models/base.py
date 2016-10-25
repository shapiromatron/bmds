import os
import tempfile

from .. import constants
from ..parser import OutputParser
from ..utils import RunProcess


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bin'))


class BMDModel(object):

    def __init__(self, dataset, overrides=None, id=None):
        self.id = id
        self.dataset = dataset
        self.overrides = overrides or {}
        self.values = {}
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
                with open(outfile, 'r') as f:
                    text = f.read()
                self.parse_results(text)
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
    def get_default(cls):
        return {
            'name': cls.model_name,
            'defaults': cls.defaults
        }

    @classmethod
    def get_exe_path(cls):
        return os.path.abspath(os.path.join(
            ROOT,
            cls.bmds_version_dir,
            cls.exe + '.exe'))

    def parse_results(self, outfile):
        parser = OutputParser(outfile, self.dtype, self.model_name)
        self.outfile = outfile
        self.output = parser.output

    def as_dfile(self):
        raise NotImplementedError('Abstract method requires implementation')

    def write_dfile(self):
        f_in = self.get_tempfile(suffix='.(d)')
        with open(f_in, 'w') as f:
            f.write(self.as_dfile())
        return f_in

    def _set_values(self):
        self.values = {}
        for k in self.defaults.keys():
            self.values[k] = self._get_option_value(k)

    def _get_option_value(self, key):
        """
        Get option value(s), or use default value if no override value.

        Two output values for 'p' type values (parameters), else one.
        Returns a single value or tuple of two values
        """
        val = self.overrides[key] \
            if key in self.overrides \
            else self.defaults[key]['d']

        if self.defaults[key]['t'] == constants.FT_PARAM:
            return val.split('|')
        else:
            return val

    def _dfile_print_header_rows(self):
        return '{}\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out'.format(self.model_name)  # noqa

    def _dfile_print_parameters(self, *params):
        # Print parameters in the specified order. Expects a tuple of parameter
        # names, in the proper order.
        if ((self.dtype == constants.CONTINUOUS) and
                (self.values['constant_variance'] == 1)):
            self.values['rho'] = ('s', 0)  # for specified to equal 0
        specifieds = []
        initials = []
        init = '0'  # 1 if initialized, 0 otherwise
        for param in params:
            t, v = self.values[param]
            # now add values
            if t == constants.P_DEFAULT:
                specifieds.append(-9999)
                initials.append(-9999)
            elif t == constants.P_SPECIFIED:
                specifieds.append(v)
                initials.append(-9999)
            elif t == constants.P_INITIALIZED:
                init = '1'
                specifieds.append(-9999)
                initials.append(v)
            else:
                raise ValueError('Unknown parameter specification')

        return '\n'.join([
            ' '.join([str(i) for i in specifieds]),
            init,
            ' '.join([str(i) for i in initials])
        ])

    def _dfile_print_options(self, *params):
        # Return space-separated list of values for dfile
        return ' '.join([str(self.values[param]) for param in params])

    def _get_model_name(self):
        return self.exe


class DefaultParams(object):
    """
    Container to store default modeling input parameters.

    Key crosswalk:
     - c = category
     - t = field type
     - d = default
     - n = name (optional)
    """

    bmdl_curve_calculation = {
        'c': constants.FC_OTHER,
        't': constants.FT_BOOL,
        'd': 0,
    }
    append_or_overwrite = {
        'c': constants.FC_OTHER,
        't': constants.FT_BOOL,
        'd': 0,
    }
    smooth_option = {
        'c': constants.FC_OTHER,
        't': constants.FT_BOOL,
        'd': 0
    }
    bmd_calculation = {
        'c': constants.FC_OTHER,
        't': constants.FT_BOOL,
        'd': 1,
        'n': 'BMD calculation',
    }
    bmdl_curve_calc = {
        'c': constants.FC_OTHER,
        't': constants.FT_BOOL,
        'd': 0,
        'n': 'BMDL curve calculation',
    }
    dose_drop = {
        'c': constants.FC_OTHER,
        't': constants.FT_DROPDOSE,
        'd': 0,
        'n': 'Doses to drop',
    }
    constant_variance = {
        'c': constants.FC_OTHER,
        't': constants.FT_BOOL,
        'd': 1,
        'n': 'Constant variance',
    }
    max_iterations = {
        'c': constants.FC_OPTIMIZER,
        't': constants.FT_INTEGER,
        'd': 250,
        'n': 'Iteration',
    }
    relative_fn_conv = {
        'c': constants.FC_OPTIMIZER,
        't': constants.FT_DECIMAL,
        'd': 1.0E-08,
        'n': 'Relative function',
    }
    parameter_conv = {
        'c': constants.FC_OPTIMIZER,
        't': constants.FT_DECIMAL,
        'd': 1.0E-08,
        'n': 'Parameter',
    }
    confidence_level = {
        'c': constants.FC_BMR,
        't': constants.FT_DECIMAL,
        'd': 0.95,
    }
    dich_bmr = {
        'c': constants.FC_BMR,
        't': constants.FT_INTEGER,
        'd': 0.1,
    }
    dich_bmr_type = {
        'c': constants.FC_BMR,
        't': constants.FT_DECIMAL,
        'd': 0,
    }
    cont_bmr = {
        'c': constants.FC_BMR,
        't': constants.FT_DECIMAL,
        'd': 1.0,
    }
    cont_bmr_type = {
        'c': constants.FC_BMR,
        't': constants.FT_INTEGER,
        'd': 1,
    }

    @staticmethod
    def degree_poly(d=2, showName=True):
        d = {
            'c': constants.FC_OTHER,
            't': constants.FT_INTEGER,
            'd': d,
        }
        if showName:
            d['n'] = 'Degree of polynomial'
        return d

    @staticmethod
    def log_transform(d):
        return {
            'c': constants.FC_OTHER,
            't': constants.FT_BOOL,
            'd': d,
        }

    @staticmethod
    def param_generator(name=None):
        d = {
            'c': constants.FC_PARAM,
            't': constants.FT_PARAM,
            'd': 'd|',
        }
        if name:
            d['n'] = name
        return d

    @staticmethod
    def restrict(d, n=None):
        d = {
            'c': constants.FC_OTHER,
            't': constants.FT_BOOL,
            'd': d,
        }
        if n:
            d['n'] = n
        return d
