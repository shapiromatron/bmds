from .base import BMDModel


class Polynomial_216(BMDModel):

    def dfile_print(self, dataset):
        """Custom file for printing dfile, using helper functions for BMD
        parent class."""
        txt = self._dfile_print_header()
        degpoly = int(self.values['degree_poly'][0])
        txt.append(str(degpoly))
        txt.append('1 ' + str(dataset['numDG'] - self.values['dose_drop'][0]) + ' 0')
        p = ('max_iterations', 'relative_fn_conv', 'parameter_conv',
             'bmdl_curve_calculation', 'restrict_polynomial',
             'bmd_calculation', 'append_or_overwrite', 'smooth_option')
        txt.append(self._dfile_print_options(p))
        p = ('bmr_type', 'bmr',  'constant_variance', 'confidence_level')
        txt.append(self._dfile_print_options(p))
        p = ['alpha', 'rho', 'beta_0']
        for i in xrange(1, degpoly + 1):
            p.append('beta_' + str(i))
        txt.append(self._dfile_print_parameters(p))
        txt.append(self._dfile_print_continuous_dataset(dataset))
        return '\n'.join(txt)

    #todo: add check that degree poly must be <=8
    minimum_DG = 2
    model_name = 'Polynomial'
    dtype = 'C'
    exe = 'poly'
    exe_plot = '00poly'
    js_formula = "{beta_0} + ({beta_1}*x) + ({beta_2}*Math.pow(x,2)) + ({beta_3}*Math.pow(x,3)) + ({beta_4}*Math.pow(x,4)) + ({beta_5}*Math.pow(x,5)) + ({beta_6}*Math.pow(x,6)) + ({beta_7}*Math.pow(x,7)) + ({beta_8}*Math.pow(x,8))"
    js_parameters = ['beta_0', 'beta_1', 'beta_2', 'beta_3', 'beta_4', 'beta_5', 'beta_6', 'beta_7', 'beta_8']
    version = 2.16
    date = '05/26/2010'
    defaults = {
        'bmdl_curve_calculation':   {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'append_or_overwrite':      {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'smooth_option':            {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'max_iterations':           {'c': 'op', 't': 'i', 'f': 0, 'd': 250, 'n': 'Iteration'},
        'relative_fn_conv':         {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Relative Function'},
        'parameter_conv':           {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Parameter'},
        'alpha':                    {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Alpha'},
        'rho':                      {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Rho'},
        'beta_0':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta0'},
        'beta_1':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta1'},
        'beta_2':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta2'},
        'beta_3':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta3'},
        'beta_4':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta4'},
        'beta_5':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta5'},
        'beta_7':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta7'},
        'beta_6':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta6'},
        'beta_8':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta8'},
        'restrict_polynomial':      {'c': 'ot', 't': 'rp', 'f': 0, 'd': 0, 'n': 'Restrict Polynomial'},
        'degree_poly':              {'c': 'ot', 't': 'i', 'f': 0, 'd': 2, 'n': 'Degree of Polynomial'},
        'bmd_calculation':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'BMD Calculation'},
        'bmdl_curve_calc':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 0, 'n': 'BMDL Curve Calculation'},
        'dose_drop':                {'c': 'ot', 't': 'dd', 'f': 0, 'd': 0, 'n': 'Doses to drop'},
        'bmr':                      {'c': 'b',  't': 'd', 'f': 1, 'd': 1.0},
        'bmr_type':                 {'c': 'b',  't': 'i', 'f': 1, 'd': 1},
        'confidence_level':         {'c': 'b',  't': 'd', 'f': 1, 'd': 0.95},
        'constant_variance':        {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'Constant Variance'}}
    possible_bmr = ('Abs. Dev.', 'Std. Dev.', 'Rel. Dev.', 'Point', 'Extra')


class Polynomial_217(Polynomial_216):
    version = 2.17
    date = '01/28/2013'
    defaults = Polynomial_216.defaults.copy()
    defaults['max_iterations']['d'] = 500


class Linear_216(BMDModel):
    """ Overrides of Polynomial for Linear model. """

    def dfile_print(self, dataset):
        """Custom file for printing dfile, using helper functions for BMD
        parent class."""
        txt = self._dfile_print_header()
        txt.append('1')
        txt.append('1 ' + str(dataset['numDG'] - self.values['dose_drop'][0]) + ' 0')
        p = ('max_iterations', 'relative_fn_conv', 'parameter_conv',
             'bmdl_curve_calculation', 'restrict_polynomial',
             'bmd_calculation', 'append_or_overwrite', 'smooth_option')
        txt.append(self._dfile_print_options(p))
        p = ('bmr_type', 'bmr',  'constant_variance', 'confidence_level')
        txt.append(self._dfile_print_options(p))
        p = ['alpha', 'rho', 'beta_0', 'beta_1']
        txt.append(self._dfile_print_parameters(p))
        txt.append(self._dfile_print_continuous_dataset(dataset))
        return '\n'.join(txt)

    #todo: add check that degree poly must be <=8
    minimum_DG = 2
    model_name = 'Linear'
    dtype = 'C'
    exe = 'poly'
    exe_plot = '00poly'
    js_formula = "{beta_0} + ({beta_1}*x)"
    js_parameters = ['beta_0', 'beta_1']
    version = 2.16
    date = '05/26/2010'
    defaults = {
        'bmdl_curve_calculation':   {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'append_or_overwrite':      {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'smooth_option':            {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'max_iterations':           {'c': 'op', 't': 'i', 'f': 0, 'd': 250, 'n': 'Iteration'},
        'relative_fn_conv':         {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Relative Function'},
        'parameter_conv':           {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Parameter'},
        'alpha':                    {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Alpha'},
        'rho':                      {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Rho'},
        'beta_0':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta0'},
        'beta_1':                   {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Beta1'},
        'restrict_polynomial':      {'c': 'ot', 't': 'rp', 'f': 0, 'd': 0, 'n': 'Restrict Polynomial'},
        'degree_poly':              {'c': 'ot', 't': 'i', 'f': 1, 'd': 1},
        'bmd_calculation':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'BMD Calculation'},
        'bmdl_curve_calc':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 0, 'n': 'BMDL Curve Calculation'},
        'dose_drop':                {'c': 'ot', 't': 'dd', 'f': 0, 'd': 0, 'n': 'Doses to drop'},
        'bmr':                      {'c': 'b',  't': 'd', 'f': 1, 'd': 1.0},
        'bmr_type':                 {'c': 'b',  't': 'i', 'f': 1, 'd': 1},
        'confidence_level':         {'c': 'b',  't': 'd', 'f': 1, 'd': 0.95},
        'constant_variance':        {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'Constant Variance'}}
    possible_bmr = ('Abs. Dev.', 'Std. Dev.', 'Rel. Dev.', 'Point', 'Extra')


class Linear_217(Linear_216):
    version = 2.17
    date = '01/28/2013'
    defaults = Linear_216.defaults.copy()
    defaults['max_iterations']['d'] = 500


class Exponential_M2_17(BMDModel):

    def dfile_print(self, dataset):
        """
        Custom function for printing exponential dfiles.
        """
        txt = self._dfile_print_header()
        txt.append('1 ' + str(dataset['numDG'] - self.values['dose_drop'][0]) + self.exp_run_settings)
        p = ('max_iterations', 'relative_fn_conv', 'parameter_conv',
             'bmdl_curve_calculation', 'bmd_calculation',
             'append_or_overwrite', 'smooth_option')
        txt.append(self._dfile_print_options(p))
        p = ('bmr_type', 'bmr', 'constant_variance', 'confidence_level')
        txt.append(self._dfile_print_options(p))
        p = ('alpha', 'rho', 'a', 'b', 'c', 'd')
        v = self._dfile_print_parameters(p)
        txt.append('\n'.join([v for i in xrange(4)]))
        txt.append(self._dfile_print_continuous_dataset(dataset))
        return '\n'.join(txt)

    minimum_DG = 2
    pretty_name = 'Exponential-M2'
    model_name = 'Exponential'
    dtype = 'C'
    exe = 'exponential'
    exe_plot = 'Expo_CPlot'
    js_formula = "{a} * Math.exp({sign}*{b}*x)"
    exp_run_settings = ' 0 1000 11 0 1'
    js_parameters = ['a', 'b', 'sign']
    version = 1.7
    date = '12/10/2009'
    defaults = {
        'bmdl_curve_calculation':   {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'append_or_overwrite':      {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'smooth_option':            {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'max_iterations':           {'c': 'op', 't': 'i', 'f': 0, 'd': 250, 'n': 'Iteration'},
        'relative_fn_conv':         {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Relative Function'},
        'parameter_conv':           {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Parameter'},
        'alpha':                    {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Alpha'},
        'rho':                      {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Rho'},
        'a':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'a'},
        'b':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'b'},
        'c':                        {'c': 'p',  't': 'p', 'f': 1, 'd': 'd|', 'n': 'c'},
        'd':                        {'c': 'p',  't': 'p', 'f': 1, 'd': 'd|', 'n': 'd'},
        'bmd_calculation':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'BMD Calculation'},
        'bmdl_curve_calc':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 0, 'n': 'BMDL Curve Calculation'},
        'dose_drop':                {'c': 'ot', 't': 'dd', 'f': 0, 'd': 0, 'n': 'Doses to drop'},
        'bmr':                      {'c': 'b',  't': 'd', 'f': 1, 'd': 1.0},
        'bmr_type':                 {'c': 'b',  't': 'i', 'f': 1, 'd': 1},
        'confidence_level':         {'c': 'b',  't': 'd', 'f': 1, 'd': 0.95},
        'constant_variance':        {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'Constant Variance'}}
    possible_bmr = ('Abs. Dev.', 'Std. Dev.', 'Rel. Dev.', 'Point', 'Extra')
    output_prefix = 'M2'


class Exponential_M2_19(Exponential_M2_17):
    version = 1.9
    date = '01/29/2013'
    defaults = Exponential_M2_17.defaults.copy()
    defaults['max_iterations']['d'] = 500


class Exponential_M3_17(BMDModel):

    def dfile_print(self, dataset):
        """
        Custom function for printing exponential dfiles.
        """
        txt = self._dfile_print_header()
        txt.append('1 ' + str(dataset['numDG'] - self.values['dose_drop'][0]) + self.exp_run_settings)
        p = ('max_iterations', 'relative_fn_conv', 'parameter_conv',
             'bmdl_curve_calculation', 'bmd_calculation',
             'append_or_overwrite', 'smooth_option')
        txt.append(self._dfile_print_options(p))
        p = ('bmr_type', 'bmr', 'constant_variance', 'confidence_level')
        txt.append(self._dfile_print_options(p))
        p = ('alpha', 'rho', 'a', 'b', 'c', 'd')
        v = self._dfile_print_parameters(p)
        txt.append('\n'.join([v for i in xrange(4)]))
        txt.append(self._dfile_print_continuous_dataset(dataset))
        return '\n'.join(txt)

    minimum_DG = 3
    pretty_name = 'Exponential-M3'
    model_name = 'Exponential'
    dtype = 'C'
    exe = 'exponential'
    exe_plot = 'Expo_CPlot'
    js_formula = "{a} * Math.exp({sign}*Math.pow({b}*x,{d}))"
    exp_run_settings = ' 0 0100 22 0 1'
    js_parameters = ['a', 'b', 'd', 'sign']
    version = 1.7
    date = '12/10/2009'
    defaults = {
        'bmdl_curve_calculation':   {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'append_or_overwrite':      {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'smooth_option':            {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'max_iterations':           {'c': 'op', 't': 'i', 'f': 0, 'd': 250, 'n': 'Iteration'},
        'relative_fn_conv':         {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Relative Function'},
        'parameter_conv':           {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Parameter'},
        'alpha':                    {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Alpha'},
        'rho':                      {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Rho'},
        'a':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'a'},
        'b':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'b'},
        'c':                        {'c': 'p',  't': 'p', 'f': 1, 'd': 'd|', 'n': 'c'},
        'd':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'd'},
        'bmd_calculation':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'BMD Calculation'},
        'bmdl_curve_calc':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 0, 'n': 'BMDL Curve Calculation'},
        'dose_drop':                {'c': 'ot', 't': 'dd', 'f': 0, 'd': 0, 'n': 'Doses to drop'},
        'bmr':                      {'c': 'b',  't': 'd', 'f': 1, 'd': 1.0},
        'bmr_type':                 {'c': 'b',  't': 'i', 'f': 1, 'd': 1},
        'confidence_level':         {'c': 'b',  't': 'd', 'f': 1, 'd': 0.95},
        'constant_variance':        {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'Constant Variance'}}
    possible_bmr = ('Abs. Dev.', 'Std. Dev.', 'Rel. Dev.', 'Point', 'Extra')
    output_prefix = 'M3'


class Exponential_M3_19(Exponential_M3_17):
    version = 1.9
    ddate = '01/29/2013'
    defaults = Exponential_M3_17.defaults.copy()
    defaults['max_iterations']['d'] = 500


class Exponential_M4_17(BMDModel):

    def dfile_print(self, dataset):
        """
        Custom function for printing exponential dfiles.
        """
        txt = self._dfile_print_header()
        txt.append('1 ' + str(dataset['numDG'] - self.values['dose_drop'][0]) + self.exp_run_settings)
        p = ('max_iterations', 'relative_fn_conv', 'parameter_conv',
             'bmdl_curve_calculation', 'bmd_calculation',
             'append_or_overwrite', 'smooth_option')
        txt.append(self._dfile_print_options(p))
        p = ('bmr_type', 'bmr', 'constant_variance', 'confidence_level')
        txt.append(self._dfile_print_options(p))
        p = ('alpha', 'rho', 'a', 'b', 'c', 'd')
        v = self._dfile_print_parameters(p)
        txt.append('\n'.join([v for i in xrange(4)]))
        txt.append(self._dfile_print_continuous_dataset(dataset))
        return '\n'.join(txt)

    minimum_DG = 3
    pretty_name = 'Exponential-M4'
    model_name = 'Exponential'
    dtype = 'C'
    exe = 'exponential'
    exe_plot = 'Expo_CPlot'
    js_formula = "{a} * ({c}-({c}-1) * Math.exp(-1.*{b}*x))"
    exp_run_settings = ' 0 0010 33 0 1'
    js_parameters = ['a', 'b', 'c']
    version = 1.7
    date = '12/10/2009'
    defaults = {
        'bmdl_curve_calculation':   {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'append_or_overwrite':      {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'smooth_option':            {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'max_iterations':           {'c': 'op', 't': 'i', 'f': 0, 'd': 250, 'n': 'Iteration'},
        'relative_fn_conv':         {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Relative Function'},
        'parameter_conv':           {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Parameter'},
        'alpha':                    {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Alpha'},
        'rho':                      {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Rho'},
        'a':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'a'},
        'b':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'b'},
        'c':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'c'},
        'd':                        {'c': 'p',  't': 'p', 'f': 1, 'd': 'd|', 'n': 'd'},
        'bmd_calculation':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'BMD Calculation'},
        'bmdl_curve_calc':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 0, 'n': 'BMDL Curve Calculation'},
        'dose_drop':                {'c': 'ot', 't': 'dd', 'f': 0, 'd': 0, 'n': 'Doses to drop'},
        'bmr':                      {'c': 'b',  't': 'd', 'f': 1, 'd': 1.0},
        'bmr_type':                 {'c': 'b',  't': 'i', 'f': 1, 'd': 1},
        'confidence_level':         {'c': 'b',  't': 'd', 'f': 1, 'd': 0.95},
        'constant_variance':        {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'Constant Variance'}}
    possible_bmr = ('Abs. Dev.', 'Std. Dev.', 'Rel. Dev.', 'Point', 'Extra')
    output_prefix = 'M4'


class Exponential_M4_19(Exponential_M4_17):
    version = 1.9
    date = '01/29/2013'
    defaults = Exponential_M4_17.defaults.copy()
    defaults['max_iterations']['d'] = 500


class Exponential_M5_17(BMDModel):

    def dfile_print(self, dataset):
        """
        Custom function for printing exponential dfiles.
        """
        txt = self._dfile_print_header()
        txt.append('1 ' + str(dataset['numDG'] - self.values['dose_drop'][0]) + self.exp_run_settings)
        p = ('max_iterations', 'relative_fn_conv', 'parameter_conv',
             'bmdl_curve_calculation', 'bmd_calculation',
             'append_or_overwrite', 'smooth_option')
        txt.append(self._dfile_print_options(p))
        p = ('bmr_type', 'bmr', 'constant_variance', 'confidence_level')
        txt.append(self._dfile_print_options(p))
        p = ('alpha', 'rho', 'a', 'b', 'c', 'd')
        v = self._dfile_print_parameters(p)
        txt.append('\n'.join([v for i in xrange(4)]))
        txt.append(self._dfile_print_continuous_dataset(dataset))
        return '\n'.join(txt)

    minimum_DG = 4
    pretty_name = 'Exponential-M5'
    model_name = 'Exponential'
    dtype = 'C'
    exe = 'exponential'
    exe_plot = 'Expo_CPlot'
    js_formula = "{a} * ({c}-({c}-1) *  Math.exp(-1.*Math.pow({b}*x,{d})))"
    exp_run_settings = ' 0 0001 44 0 1'
    js_parameters = ['a', 'b', 'c', 'd']
    version = 1.7
    date = '12/10/2009'
    defaults = {
        'bmdl_curve_calculation':   {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'append_or_overwrite':      {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'smooth_option':            {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'max_iterations':           {'c': 'op', 't': 'i', 'f': 0, 'd': 250, 'n': 'Iteration'},
        'relative_fn_conv':         {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Relative Function'},
        'parameter_conv':           {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Parameter'},
        'alpha':                    {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Alpha'},
        'rho':                      {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Rho'},
        'a':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'a'},
        'b':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'b'},
        'c':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'c'},
        'd':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'd'},
        'bmd_calculation':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'BMD Calculation'},
        'bmdl_curve_calc':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 0, 'n': 'BMDL Curve Calculation'},
        'dose_drop':                {'c': 'ot', 't': 'dd', 'f': 0, 'd': 0, 'n': 'Doses to drop'},
        'bmr':                      {'c': 'b',  't': 'd', 'f': 1, 'd': 1.0},
        'bmr_type':                 {'c': 'b',  't': 'i', 'f': 1, 'd': 1},
        'confidence_level':         {'c': 'b',  't': 'd', 'f': 1, 'd': 0.95},
        'constant_variance':        {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'Constant Variance'}}
    possible_bmr = ('Abs. Dev.', 'Std. Dev.', 'Rel. Dev.', 'Point', 'Extra')
    output_prefix = 'M5'


class Exponential_M5_19(Exponential_M5_17):
    version = 1.9
    date = '01/29/2013'
    defaults = Exponential_M5_17.defaults.copy()
    defaults['max_iterations']['d'] = 500


class Power_216(BMDModel):
    def dfile_print(self, dataset):
        """Custom file for printing dfile, using helper functions for BMD
        parent class."""
        txt = self._dfile_print_header()
        txt.append('1 ' + str(dataset['numDG'] - self.values['dose_drop'][0]) + ' 0')
        p = ('max_iterations', 'relative_fn_conv', 'parameter_conv',
             'bmdl_curve_calculation', 'restrict_power',
             'bmd_calculation', 'append_or_overwrite', 'smooth_option')
        txt.append(self._dfile_print_options(p))
        p = ('bmr_type', 'bmr', 'constant_variance', 'confidence_level')
        txt.append(self._dfile_print_options(p))
        p = ('alpha', 'rho', 'control', 'slope', 'power')
        txt.append(self._dfile_print_parameters(p))
        txt.append(self._dfile_print_continuous_dataset(dataset))
        return '\n'.join(txt)

    minimum_DG = 3
    model_name = 'Power'
    dtype = 'C'
    exe = 'power'
    exe_plot = '00power'
    js_formula = "{control} + {slope} * Math.pow(x,{power})"
    js_parameters = ['control', 'slope', 'power']
    version = 2.16
    date = '10/28/2009'
    defaults = {
        'bmdl_curve_calculation':   {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'append_or_overwrite':      {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'smooth_option':            {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'log_transform':            {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'max_iterations':           {'c': 'op', 't': 'i', 'f': 0, 'd': 250, 'n': 'Iteration'},
        'relative_fn_conv':         {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Relative Function'},
        'parameter_conv':           {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Parameter'},
        'alpha':                    {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Alpha'},
        'rho':                      {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Rho'},
        'control':                  {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Control'},
        'slope':                    {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Slope'},
        'power':                    {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Power'},
        'restrict_power':           {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'Restrict Power'},
        'bmd_calculation':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'BMD Calculation'},
        'bmdl_curve_calc':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 0, 'n': 'BMDL Curve Calculation'},
        'dose_drop':                {'c': 'ot', 't': 'dd', 'f': 0, 'd': 0, 'n': 'Doses to drop'},
        'bmr':                      {'c': 'b',  't': 'd', 'f': 1, 'd': 1.0},
        'bmr_type':                 {'c': 'b',  't': 'i', 'f': 1, 'd': 1},
        'confidence_level':         {'c': 'b',  't': 'd', 'f': 1, 'd': 0.95},
        'constant_variance':        {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'Constant Variance'}}
    possible_bmr = ('Abs. Dev.', 'Std. Dev.', 'Rel. Dev.', 'Point', 'Extra')


class Power_217(Power_216):
    version = 2.17
    date = '01/28/2013'
    defaults = Power_216.defaults.copy()
    defaults['max_iterations']['d'] = 500


class Hill_216(BMDModel):
    def dfile_print(self, dataset):
        """Custom file for printing dfile, using helper functions for BMD
        parent class."""
        txt = self._dfile_print_header()
        txt.append('1 ' + str(dataset['numDG'] - self.values['dose_drop'][0]) + ' 0')
        p = ('max_iterations', 'relative_fn_conv', 'parameter_conv',
             'bmdl_curve_calculation', 'restrict_n',
             'bmd_calculation', 'append_or_overwrite', 'smooth_option')
        txt.append(self._dfile_print_options(p))
        p = ('bmr_type', 'bmr', 'constant_variance', 'confidence_level')
        txt.append(self._dfile_print_options(p))
        p = ('alpha', 'rho', 'intercept', 'v', 'n', 'k')
        txt.append(self._dfile_print_parameters(p))
        txt.append(self._dfile_print_continuous_dataset(dataset))
        return '\n'.join(txt)

    minimum_DG = 4
    model_name = 'Hill'
    dtype = 'C'
    exe = 'hill'
    exe_plot = '00Hill'
    js_formula = "{intercept} + ({v}*Math.pow(x,{n})) / (Math.pow({k},{n}) + Math.pow(x,{n}))"
    js_parameters = ['intercept', 'v', 'n', 'k']
    version = 2.16
    date = '04/06/2011'
    defaults = {
        'bmdl_curve_calculation':   {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'append_or_overwrite':      {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'smooth_option':            {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'log_transform':            {'c': 'ot', 't': 'b', 'f': 1, 'd': 0},
        'max_iterations':           {'c': 'op', 't': 'i', 'f': 0, 'd': 250, 'n': 'Iteration'},
        'relative_fn_conv':         {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Relative Function'},
        'parameter_conv':           {'c': 'op', 't': 'd', 'f': 0, 'd': 1.0E-08, 'n': 'Parameter'},
        'alpha':                    {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Alpha'},
        'rho':                      {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Rho'},
        'intercept':                {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'Intercept'},
        'v':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'V'},
        'n':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'N'},
        'k':                        {'c': 'p',  't': 'p', 'f': 0, 'd': 'd|', 'n': 'K'},
        'restrict_n':               {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'Restrict N>1'},
        'bmd_calculation':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'BMD Calculation'},
        'bmdl_curve_calc':          {'c': 'ot', 't': 'b', 'f': 0, 'd': 0, 'n': 'BMDL Curve Calculation'},
        'dose_drop':                {'c': 'ot', 't': 'dd', 'f': 0, 'd': 0, 'n': 'Doses to drop'},
        'bmr':                      {'c': 'b',  't': 'd', 'f': 1, 'd': 1.0},
        'bmr_type':                 {'c': 'b',  't': 'i', 'f': 1, 'd': 1},
        'confidence_level':         {'c': 'b',  't': 'd', 'f': 1, 'd': 0.95},
        'constant_variance':        {'c': 'ot', 't': 'b', 'f': 0, 'd': 1, 'n': 'Constant Variance'}}
    possible_bmr = ('Abs. Dev.', 'Std. Dev.', 'Rel. Dev.', 'Point', 'Extra')


class Hill_217(Hill_216):
    version = 2.17
    date = '01/28/2013'
    defaults = Hill_216.defaults.copy()
    defaults['max_iterations']['d'] = 500
