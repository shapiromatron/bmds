from copy import deepcopy

from .base import BMDModel, DefaultParams
from .. import constants


class Continuous(BMDModel):
    possible_bmr = ('Abs. Dev.', 'Std. Dev.', 'Rel. Dev.', 'Point', 'Extra')
    dtype = constants.CONTINUOUS


class Polynomial_216(Continuous):
    # todo: add check that degree poly must be <=8
    minimum_DG = 2
    model_name = 'Polynomial'
    bmds_version_dir = 'BMDS231'
    exe = 'poly'
    exe_plot = '00poly'
    version = 2.16
    date = '05/26/2010'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'alpha': DefaultParams.param_generator('Alpha'),
        'rho': DefaultParams.param_generator('Rho'),
        'beta_0': DefaultParams.param_generator('Beta0'),
        'beta_1': DefaultParams.param_generator('Beta1'),
        'beta_2': DefaultParams.param_generator('Beta2'),
        'beta_3': DefaultParams.param_generator('Beta3'),
        'beta_4': DefaultParams.param_generator('Beta4'),
        'beta_5': DefaultParams.param_generator('Beta5'),
        'beta_7': DefaultParams.param_generator('Beta7'),
        'beta_6': DefaultParams.param_generator('Beta6'),
        'beta_8': DefaultParams.param_generator('Beta8'),
        'restrict_polynomial': DefaultParams.restrict(d=0, n='Restrict polynomial'),  # noqa
        'degree_poly': DefaultParams.degree_poly(),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.cont_bmr,
        'bmr_type': DefaultParams.cont_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
        'constant_variance': DefaultParams.constant_variance,
    }

    def as_dfile(self):
        self._set_values()
        degpoly = int(self.values['degree_poly'])
        params = ['alpha', 'rho', 'beta_0']
        for i in range(1, degpoly + 1):
            params.append('beta_' + str(i))

        return '\n'.join([
            self._dfile_print_header_rows(),
            str(degpoly),
            '1 {} 0'.format(self.dataset.doses_used),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'restrict_polynomial',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr_type', 'bmr', 'constant_variance', 'confidence_level'),
            self._dfile_print_parameters(*params),
            self.dataset.as_dfile(),
        ])


class Polynomial_217(Polynomial_216):
    bmds_version_dir = 'BMDS240'
    version = 2.17
    date = '01/28/2013'
    defaults = deepcopy(Polynomial_216.defaults)
    defaults['max_iterations']['d'] = 500


class Polynomial_220(Polynomial_217):
    bmds_version_dir = 'BMDS260'
    version = 2.20
    date = '10/22/2014'


class Linear_216(Polynomial_216):
    # todo: add check that degree poly must be <=8
    minimum_DG = 2
    model_name = 'Linear'
    bmds_version_dir = 'BMDS231'
    exe = 'poly'
    exe_plot = '00poly'
    version = 2.16
    date = '05/26/2010'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'alpha': DefaultParams.param_generator('Alpha'),
        'rho': DefaultParams.param_generator('Rho'),
        'beta_0': DefaultParams.param_generator('Beta0'),
        'beta_1': DefaultParams.param_generator('Beta1'),
        'restrict_polynomial': DefaultParams.restrict(d=0, n='Restrict polynomial'),  # noqa
        'degree_poly': DefaultParams.degree_poly(d=1, showName=False),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.cont_bmr,
        'bmr_type': DefaultParams.cont_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
        'constant_variance': DefaultParams.constant_variance,
    }

    def as_dfile(self):
        self._set_values()
        return '\n'.join([
            self._dfile_print_header_rows(),
            '1',
            '1 {} 0'.format(self.dataset.doses_used),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'restrict_polynomial',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr_type', 'bmr', 'constant_variance', 'confidence_level'),
            self._dfile_print_parameters(
                'alpha', 'rho', 'beta_0', 'beta_1'),
            self.dataset.as_dfile(),
        ])


class Linear_217(Linear_216):
    bmds_version_dir = 'BMDS240'
    version = 2.17
    date = '01/28/2013'
    defaults = deepcopy(Linear_216.defaults)
    defaults['max_iterations']['d'] = 500


class Linear_220(Linear_217):
    bmds_version_dir = 'BMDS260'
    version = 2.20
    date = '10/22/2014'


class Exponential(Continuous):

    def _get_model_name(self):
        return '{}_{}'.format(self.exe, self.output_prefix.lower())

    def as_dfile(self):
        self._set_values()
        params = self._dfile_print_parameters(
            'alpha', 'rho', 'a', 'b', 'c', 'd')
        return '\n'.join([
            self._dfile_print_header_rows(),
            '1 {}{}'.format(self.dataset.doses_used, self.exp_run_settings),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'bmd_calculation',
                'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr_type', 'bmr', 'constant_variance', 'confidence_level'),
            '\n'.join([params] * 4),
            self.dataset.as_dfile(),
        ])


class Exponential_M2_17(Exponential):
    minimum_DG = 2
    model_name = 'Exponential-M2'
    bmds_version_dir = 'BMDS231'
    exe = 'exponential'
    exe_plot = 'Expo_CPlot'
    exp_run_settings = ' 0 1000 11 0 1'
    version = 1.7
    date = '12/10/2009'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'alpha': DefaultParams.param_generator('Alpha'),
        'rho': DefaultParams.param_generator('Rho'),
        'a': DefaultParams.param_generator('a'),
        'b': DefaultParams.param_generator('b'),
        'c': DefaultParams.param_generator('c'),
        'd': DefaultParams.param_generator('d'),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.cont_bmr,
        'bmr_type': DefaultParams.cont_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
        'constant_variance': DefaultParams.constant_variance,
    }
    output_prefix = 'M2'


class Exponential_M2_19(Exponential_M2_17):
    bmds_version_dir = 'BMDS240'
    version = 1.9
    date = '01/29/2013'
    defaults = deepcopy(Exponential_M2_17.defaults)
    defaults['max_iterations']['d'] = 500


class Exponential_M2_110(Exponential_M2_19):
    bmds_version_dir = 'BMDS260'
    version = 1.10
    date = '01/12/2015'


class Exponential_M3_17(Exponential_M2_17):
    minimum_DG = 3
    model_name = 'Exponential-M3'
    exp_run_settings = ' 0 0100 22 0 1'
    output_prefix = 'M3'


class Exponential_M3_19(Exponential_M3_17):
    bmds_version_dir = 'BMDS240'
    version = 1.9
    ddate = '01/29/2013'
    defaults = deepcopy(Exponential_M3_17.defaults)
    defaults['max_iterations']['d'] = 500


class Exponential_M3_110(Exponential_M3_19):
    bmds_version_dir = 'BMDS260'
    version = 1.10
    date = '01/12/2015'


class Exponential_M4_17(Exponential_M2_17):
    minimum_DG = 3
    model_name = 'Exponential-M4'
    exp_run_settings = ' 0 0010 33 0 1'
    output_prefix = 'M4'


class Exponential_M4_19(Exponential_M4_17):
    bmds_version_dir = 'BMDS240'
    version = 1.9
    date = '01/29/2013'
    defaults = deepcopy(Exponential_M4_17.defaults)
    defaults['max_iterations']['d'] = 500


class Exponential_M4_110(Exponential_M4_19):
    bmds_version_dir = 'BMDS260'
    version = 1.10
    date = '01/12/2015'


class Exponential_M5_17(Exponential_M2_17):
    minimum_DG = 4
    model_name = 'Exponential-M5'
    exp_run_settings = ' 0 0001 44 0 1'
    output_prefix = 'M5'


class Exponential_M5_19(Exponential_M5_17):
    bmds_version_dir = 'BMDS240'
    version = 1.9
    date = '01/29/2013'
    defaults = deepcopy(Exponential_M5_17.defaults)
    defaults['max_iterations']['d'] = 500


class Exponential_M5_110(Exponential_M5_19):
    bmds_version_dir = 'BMDS260'
    version = 1.10
    date = '01/12/2015'


class Power_216(Continuous):
    minimum_DG = 3
    model_name = 'Power'
    bmds_version_dir = 'BMDS231'
    exe = 'power'
    exe_plot = '00power'
    version = 2.16
    date = '10/28/2009'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'log_transform': DefaultParams.log_transform(d=0),
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'alpha': DefaultParams.param_generator('Alpha'),
        'rho': DefaultParams.param_generator('Rho'),
        'control': DefaultParams.param_generator('Control'),
        'slope': DefaultParams.param_generator('Slope'),
        'power': DefaultParams.param_generator('Power'),
        'restrict_power': DefaultParams.restrict(d=1, n='Restrict power'),  # noqa
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.cont_bmr,
        'bmr_type': DefaultParams.cont_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
        'constant_variance': DefaultParams.constant_variance,
    }

    def as_dfile(self):
        self._set_values()
        return '\n'.join([
            self._dfile_print_header_rows(),
            '1 {} 0'.format(self.dataset.doses_used),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'restrict_power',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr_type', 'bmr', 'constant_variance', 'confidence_level'),
            self._dfile_print_parameters(
                'alpha', 'rho', 'control', 'slope', 'power'),
            self.dataset.as_dfile(),
        ])


class Power_217(Power_216):
    bmds_version_dir = 'BMDS240'
    version = 2.17
    date = '01/28/2013'
    defaults = deepcopy(Power_216.defaults)
    defaults['max_iterations']['d'] = 500


class Power_218(Power_217):
    bmds_version_dir = 'BMDS2601'  # todo - change back after complete
    version = 2.18
    date = '05/19/2014'


class Hill_216(Continuous):
    minimum_DG = 4
    model_name = 'Hill'
    bmds_version_dir = 'BMDS231'
    exe = 'hill'
    exe_plot = '00Hill'
    version = 2.16
    date = '04/06/2011'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'log_transform': DefaultParams.log_transform(d=0),
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'alpha': DefaultParams.param_generator('Alpha'),
        'rho': DefaultParams.param_generator('Rho'),
        'intercept': DefaultParams.param_generator('Intercept'),
        'v': DefaultParams.param_generator('V'),
        'n': DefaultParams.param_generator('N'),
        'k': DefaultParams.param_generator('K'),
        'restrict_n': DefaultParams.restrict(d=1, n='Restrict N>1'),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.cont_bmr,
        'bmr_type': DefaultParams.cont_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
        'constant_variance': DefaultParams.constant_variance,
    }

    def as_dfile(self):
        self._set_values()
        return '\n'.join([
            self._dfile_print_header_rows(),
            '1 {} 0'.format(self.dataset.doses_used),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'restrict_n',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr_type', 'bmr', 'constant_variance', 'confidence_level'),
            self._dfile_print_parameters(
                'alpha', 'rho', 'intercept', 'v', 'n', 'k'),
            self.dataset.as_dfile(),
        ])


class Hill_217(Hill_216):
    bmds_version_dir = 'BMDS240'
    version = 2.17
    date = '01/28/2013'
    defaults = deepcopy(Hill_216.defaults)
    defaults['max_iterations']['d'] = 500
