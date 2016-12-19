import os
import numpy as np
import tempfile

from .. import constants, datasets
from ..parser import OutputParser
from ..utils import RunProcess


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bin'))


class TempFileMaker(object):
    # Maintains a list of temporary files and cleans up after itself

    def __init__(self):
        self.tempfns = []

    def add_tempfile(self, fn):
        self.tempfns.append(fn)

    def get_tempfile(self, prefix='', suffix='.txt'):
        fd, fn = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        os.close(fd)
        self.add_tempfile(fn)
        return fn

    def cleanup(self):
        for fn in self.tempfns:
            try:
                os.remove(fn)
            except OSError:
                pass

    def __del__(self):
        self.cleanup()


class BMDModel(TempFileMaker):

    def __init__(self, dataset, overrides=None, id=None):
        super(BMDModel, self).__init__()
        self.id = id
        self.dataset = dataset
        self.overrides = overrides or {}
        self.values = {}
        self.output_created = False

    def execute(self):
        try:
            exe = self.get_exe_path()
            dfile = self.write_dfile()
            RunProcess([exe, dfile], timeout=20).call()
            outfile = self.get_outfile(dfile)
            o2 = outfile.replace('.(d)', '.002')
            if os.path.exists(outfile):
                self.output_created = True
                self.add_tempfile(outfile)
                with open(outfile, 'r') as f:
                    text = f.read()
                self.parse_results(text)
            if os.path.exists(o2):
                self.add_tempfile(o2)
        except Exception as e:
            raise e
        finally:
            self.cleanup()

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

    @property
    def has_successfully_executed(self):
        return hasattr(self, 'outfile')

    def get_outfile(self, dfile):
        return dfile.replace('.(d)', '.out')

    def parse_results(self, outfile):
        parser = OutputParser(outfile, self.dtype, self.model_name)
        self.outfile = outfile
        self.output = parser.output

    def as_dfile(self):
        raise NotImplementedError('Abstract method requires implementation')

    def plot(self):
        plt = self.dataset.plot()
        if self.has_successfully_executed:
            plt.title(self.model_name)
            self._set_x_range(plt)
            self.add_to_plot(plt)
        else:
            self._add_plot_failure(plt)
        return plt

    def add_to_plot(self, plt):
        raise NotImplementedError('Abstract method requires implementation')

    def _add_plot_failure(self, plt):
        ax = plt.gca()
        plt.text(
            0.5, 0.8,
            u'ERROR: {} cannot be plotted'.format(self.model_name),
            style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
        )

    def _set_x_range(self, plt):
        ax = plt.gca()
        bmd = max(0, self.output['BMD'])
        bmdl = max(0, self.output['BMDL'])
        doses = self.dataset.doses

        # set values for what we'll need to calculate model curve
        min_x = min(bmd, bmdl, *doses)
        max_x = max(bmd, bmdl, *doses)
        self._xs = np.linspace(max(min_x, 1e-9), max_x, 100)

        # add a little extra padding on plot
        padding = datasets.PLOT_MARGINS * (max_x - min_x)
        ax.set_xlim(min_x - padding, max_x + padding)

    def write_dfile(self):
        f_in = self.get_tempfile(prefix='bmds-', suffix='.(d)')
        with open(f_in, 'w') as f:
            f.write(self.as_dfile())
        return f_in

    def _get_param(self, key):
        return self.output['parameters'][key]['estimate']

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
        fn_name = 'set_{}_value'.format(key)
        if key in self.overrides:
            val = self.overrides[key]
        elif hasattr(self, fn_name):
            val = getattr(self, fn_name)()
        else:
            val = self.defaults[key]['d']

        if self.defaults[key]['t'] == constants.FT_PARAM:
            return val.split('|')
        else:
            return val

    def _dfile_print_header_rows(self):
        return '{}\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out'.format(self.model_name)  # noqa

    def _dfile_print_parameters(self, *params):
        # Print parameters in the specified order. Expects a tuple of parameter
        # names, in the proper order.
        if ((self.dtype in constants.CONTINUOUS_DTYPES) and
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

    def _to_dict(self, model_index):
        return dict(
            model_index=model_index,
            model_name=self.model_name,
            model_version=self.version,
            has_output=self.output_created,
            dfile=self.as_dfile(),
            outfile=getattr(self, 'outfile', None),
            output=getattr(self, 'output', None),
            logic_bin=getattr(self, 'logic_bin', None),
            logic_notes=getattr(self, 'logic_notes', None),
            recommended=getattr(self, 'recommended', None),
            recommended_variable=getattr(self, 'recommended_variable', None),
        )



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
        'n': 'BMDL curve calculation',
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
