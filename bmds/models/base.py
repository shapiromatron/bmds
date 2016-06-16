from .. import constants


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

    def as_dfile():
        raise NotImplementedError('Abstract method requires implementation')

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
