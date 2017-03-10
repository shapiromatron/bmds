import os
import numpy as np

from .. import constants, plotting
from ..parser import OutputParser
from ..utils import RunProcess, TempFileList


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bin'))


class BMDModel(object):
    """
    Parent class for individual BMDS models.

    The interface for all models is identical to the base BMDModel class, and
    is therefore documented here.

    Example
    -------

    >>> dataset = bmds.ContinuousDataset(
            doses=[0, 10, 50, 150, 400],
            ns=[25, 25, 24, 24, 24],
            means=[2.61, 2.81, 2.96, 4.66, 11.23],
            stdevs=[0.81, 1.19, 1.37, 1.72, 2.84]
        )
    >>> model = bmds.models.Polynomial_220(
            dataset,
            overrides={"degree_poly": 3}
        )
    >>> model.execute()
    >>> model.output['BMD']
    88.3549
    """

    def __init__(self, dataset, overrides=None, id=None):
        self.tempfiles = TempFileList()
        self.id = id
        self.dataset = dataset
        self.overrides = overrides or {}
        self.values = {}
        self.output_created = False

    def execute(self):
        """
        Execute the BMDS model and parse outputs if successful.
        """

        # exit early if execution is not possible
        if not self.can_be_executed:
            self.output_created = False
            return

        try:
            exe = self.get_exe_path()
            dfile = self.write_dfile()
            RunProcess([exe, dfile], timeout=20).call()
            outfile = self.get_outfile(dfile)
            o2 = outfile.replace('.out', '.002')
            if os.path.exists(outfile):
                self.output_created = True
                self.tempfiles.append(outfile)
                with open(outfile, 'r') as f:
                    text = f.read()
                self.parse_results(text)
            if os.path.exists(o2):
                self.tempfiles.append(o2)
        except Exception as e:
            raise e
        finally:
            self.tempfiles.cleanup()

    @classmethod
    def get_default(cls):
        """
        Return default parameters for this model.
        """
        return {
            'name': cls.model_name,
            'defaults': cls.defaults
        }

    @classmethod
    def get_exe_path(cls):
        """
        Return the full path to the executable.
        """
        return os.path.abspath(os.path.join(
            ROOT,
            cls.bmds_version_dir,
            cls.exe + '.exe'))

    @property
    def can_be_executed(self):
        return self.dataset.num_dose_groups >= self.minimum_dose_groups

    @property
    def has_successfully_executed(self):
        """
        Check if model has successful completed.
        """
        return hasattr(self, 'outfile')

    @property
    def name(self):
        """
        Return the model name, and degree of polynomial for some models.
        """
        return self.model_name

    def get_outfile(self, dfile):
        return dfile.replace('.(d)', '.out')

    def parse_results(self, outfile):
        parser = OutputParser(outfile, self.dtype, self.model_name)
        self.outfile = outfile
        self.output = parser.output

    def as_dfile(self):
        """
        Represent this model in the BMDS (d) file input representation.
        """
        raise NotImplementedError('Abstract method requires implementation')

    def plot(self):
        """
        After model execution, print the dataset, curve-fit, BMD, and BMDL.

        Example
        -------

        >>> import os
        >>> fn = os.path.expanduser('~/Desktop/image.png')
        >>> fig = model.plot()
        >>> fig.savefig(fn)

        .. figure:: ../tests/resources/test_exponential_m4_plot.png
           :scale: 80%
           :align: center
           :alt: Example generated BMD plot

           BMD models can generate plots using the ``plot()`` method; an example
           is shown here.

        """
        fig = self.dataset.plot()
        ax = fig.gca()
        ax.set_title(self.name)
        if self.has_successfully_executed:
            self._set_x_range(ax)
            ax.plot(
                self._xs, self.get_ys(self._xs),
                **plotting.LINE_FORMAT)
            self._add_bmr_lines(ax)
        else:
            self._add_plot_failure(ax)
        return fig

    def get_ys(self, xs):
        raise NotImplementedError('Abstract base method; requires implementation.')

    def _add_bmr_lines(self, ax):
        # add BMD and BMDL lines to plot.
        bmd = self.output['BMD']
        bmdl = self.output['BMDL']
        xdomain = ax.xaxis.get_view_interval()
        ydomain = ax.yaxis.get_view_interval()
        xrng = xdomain[1] - xdomain[0]
        yrng = ydomain[1] - ydomain[0]
        ys = self.get_ys(np.array([bmd, bmdl]))

        ax.axhline(ys[0],
                   xmin=0,
                   xmax=(bmd - xdomain[0]) / xrng,
                   **plotting.BMD_LINE_FORMAT)
        ax.axhline(ys[1],
                   xmin=0,
                   xmax=(bmdl - xdomain[0]) / xrng,
                   **plotting.BMD_LINE_FORMAT)
        ax.axvline(bmd,
                   ymin=0,
                   ymax=(ys[0] - ydomain[0]) / yrng,
                   **plotting.BMD_LINE_FORMAT)
        ax.axvline(bmdl,
                   ymin=0,
                   ymax=(ys[1] - ydomain[0]) / yrng,
                   **plotting.BMD_LINE_FORMAT)
        ax.text(bmd + xrng * 0.01,
                ydomain[0] + yrng * 0.02,
                'BMD',
                horizontalalignment='left', **plotting.BMD_LABEL_FORMAT)
        ax.text(bmdl - xrng * 0.01,
                ydomain[0] + yrng * 0.02,
                'BMDL',
                horizontalalignment='right', **plotting.BMD_LABEL_FORMAT)

    def _add_plot_failure(self, ax):
        ax.text(
            0.5, 0.8,
            u'ERROR: model cannot be plotted',
            transform=ax.transAxes,
            **plotting.FAILURE_MESSAGE_FORMAT
        )

    def _set_x_range(self, ax):
        bmd = max(0, self.output['BMD'])
        bmdl = max(0, self.output['BMDL'])
        doses = self.dataset.doses

        # set values for what we'll need to calculate model curve
        min_x = min(bmd, bmdl, *doses)
        max_x = max(bmd, bmdl, *doses)
        self._xs = np.linspace(max(min_x, 1e-9), max_x, 100)

        # add a little extra padding on plot
        padding = plotting.PLOT_MARGINS * (max_x - min_x)
        ax.set_xlim(min_x - padding, max_x + padding)

    def write_dfile(self):
        """
        Write the generated d_file to a temporary file.
        """
        f_in = self.tempfiles.get_tempfile(prefix='bmds-', suffix='.(d)')
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

    def to_dict(self, model_index):
        """
        Return a summary of the model in a dictionary format for serialization.

        Parameters
        ----------
        model_index : int
            The index of the model in a list of models, should be unique

        Returns
        -------
        out : dictionary
            A dictionary of model inputs, and raw and parsed outputs
        """
        return dict(
            name=self.name,
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

    def _to_df(self, d, idx, show_null):

        def _nullify(show_null, value):
            return constants.NULL if show_null else value

        d['model_name'].append(_nullify(show_null, self.name))
        d['model_index'].append(_nullify(show_null, idx))
        d['model_version'].append(_nullify(show_null, self.version))
        d['has_output'].append(_nullify(show_null, self.output_created))

        # add model outputs
        outputs = {} \
            if show_null \
            else getattr(self, 'output', {})

        d['BMD'].append(outputs.get('BMD', '-'))
        d['BMDL'].append(outputs.get('BMDL', '-'))
        d['BMDU'].append(outputs.get('BMDU', '-'))
        d['CSF'].append(outputs.get('CSF', '-'))
        d['AIC'].append(outputs.get('AIC', '-'))
        d['pvalue1'].append(outputs.get('p_value1', '-'))
        d['pvalue2'].append(outputs.get('p_value2', '-'))
        d['pvalue3'].append(outputs.get('p_value3', '-'))
        d['pvalue4'].append(outputs.get('p_value4', '-'))
        d['Chi2'].append(outputs.get('Chi2', '-'))
        d['df'].append(outputs.get('df', '-'))
        d['residual_of_interest'].append(outputs.get('residual_of_interest', '-'))
        d['warnings'].append('; '.join(outputs.get('warnings', ['-'])))

        # add logic bin and warnings
        logics = getattr(self, 'logic_notes', {})
        bin_ = constants.BIN_TEXT[self.logic_bin] \
            if hasattr(self, 'logic_bin') \
            else '-'
        d['logic_bin'].append(_nullify(show_null, bin_))

        txt = '; '.join(logics.get(constants.BIN_NO_CHANGE, ['-']))
        d['logic_cautions'].append(_nullify(show_null, txt))
        txt = '; '.join(logics.get(constants.BIN_WARNING, ['-']))
        d['logic_warnings'].append(_nullify(show_null, txt))
        txt = '; '.join(logics.get(constants.BIN_FAILURE, ['-']))
        d['logic_failures'].append(_nullify(show_null, txt))

        # add recommendation and recommendation variable
        txt = getattr(self, 'recommended', '-')
        d['recommended'].append(_nullify(show_null, txt))
        txt = getattr(self, 'recommended_variable', '-')
        d['recommended_variable'].append(_nullify(show_null, txt))

        # add verbose outputs if specified
        if 'dfile' in d:
            txt = self.as_dfile()
            d['dfile'].append(_nullify(show_null, txt))
        if 'outfile' in d:
            txt = getattr(self, 'outfile', '-')
            d['outfile'].append(_nullify(show_null, txt))


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
