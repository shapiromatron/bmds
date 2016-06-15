import json
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
        - exe_plot             # wgnuplot input-file executable (without extension)
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

    def build_defaults(self, json=False):
        """
        Build default options dictionary for the BMD model, returning only
        values which can be changed the by the user.
        """
        opt = {}
        for k, v in self.defaults.iteritems():
            if v['f'] == 0:    # if not fixed
                opt[k] = v
        return json.dumps(opt) if json else opt

    def valid_bmr(self, bmr):
        #given a model instance, check if a BMR is valid for this model-type
        return bmr['type'] in self.possible_bmr

    def update_model(self, override, override_txt, bmr):
        """
        Update the override dictionary and override text dictionary,
        save new values. Also update the bmr option selected.
        """
        self.override = override
        self.override_txt = override_txt
        for k, v in self.override.iteritems():
            if str(v).find('|') == -1:
                self.values[k] = (v, False)
            else:
                self.values[k] = (v.split('|'))

        self.values['bmr_type'] = (BMR_CROSSWALK[self.dtype][bmr['type']], False)
        self.values['bmr'] = (bmr['value'], False)
        self.values['confidence_level'] = (bmr['confidence_level'], False)

    def _get_option_value(self, key):
        """
        Get option value(s), or use default value if no override value.
        Two output values for 'p' type values (parameters), else one. Returns
        a tuple of two values.
        """
        if key in self.override:
            val = self.override[key]
        else:
            val = self.defaults[key]['d']
        if self.defaults[key]['t'] == 'p':  # parameter (two values)
            return val.split('|')
        else:
            return val, False

    def _dfile_print_header(self):
        return [self.model_name, 'BMDS_Model_Run',
                '/temp/bmd/datafile.dax', '/temp/bmd/output.out']

    def _dfile_print_parameters(self, order):
        #Print parameters in the specified order. Expects a tuple of parameter
        # names, in the proper order.
        if ((self.dtype == 'C') and (self.values['constant_variance'][0] == 1)):
            self.values['rho'] = ('s', 0)  # for specified to equal 0
        specs = []
        inits = []
        init = '0'  # 1 if initialized, 0 otherwise
        for i in order:
            t, v = self.values[i]
            #now save values
            if t == 'd':
                specs.append(-9999)
                inits.append(-9999)
            elif t == 's':
                specs.append(v)
                inits.append(-9999)
            elif t == 'i':
                init = '1'
                specs.append(-9999)
                inits.append(v)
        return '\n'.join([' '.join([str(i) for i in specs]),
                          init, ' '.join([str(i) for i in inits])])

    def _dfile_print_dichotomous_dataset(self, dataset):
        # add dose-response dataset, dropping doses as specified
        dropped = self.values['dose_drop'][0]
        txt = 'Dose Incidence NEGATIVE_RESPONSE\n'
        for i, v in enumerate(dataset['dr']):
            if i < len(dataset['dr']) - dropped:
                txt += '%f %d %d\n' % (v['dose'],
                                       v['incidence'],
                                       v['n'] - v['incidence'])
        return txt

    def _dfile_print_continuous_dataset(self, dataset):
        dropped = self.values['dose_drop'][0]
        txt = 'Dose NumAnimals Response Stdev\n'
        for i, v in enumerate(dataset['dr']):
            if i < len(dataset['dr']) - dropped:
                txt += '%f %f %f %f\n' % (v['dose'],
                                          v['n'],
                                          v['response'],
                                          v['stdev'])
        return txt

    def _dfile_print_options(self, order):
        #helper function; given tuple order of parameters in the 'value'
        # dictionary, return a space-separated list
        r = []
        for f in order:
            r.append(self.values[f][0])
        return ' '.join([str(i) for i in r])

    def __init__(self):
        #save default values originally
        self.values = {}
        self.override = {}
        self.override_txt = ['']
        for k, v in self.defaults.iteritems():
            self.values[k] = self._get_option_value(k)
