from copy import deepcopy

from .base import BMDModel, DefaultParams
from .. import constants


class Dichotomous(BMDModel):
    dtype = constants.DICHOTOMOUS
    possible_bmr = ('Extra', 'Added')


class DichotomousCancer(Dichotomous):
    dtype = constants.DICHOTOMOUS_CANCER
    possible_bmr = ('Extra', 'Added')


class Multistage_32(Dichotomous):
    # todo: add check that degree poly must be <=8
    minimum_DG = 2
    model_name = 'Multistage'
    bmds_version_dir = 'v231'
    exe = 'multistage'
    exe_plot = '10multista'
    js_formula = "{Background} + (1. - {Background}) * (1. - Math.exp( -1. * {Beta(1)}*x - {Beta(2)}*Math.pow(x,2) - {Beta(3)}*Math.pow(x,3) - {Beta(4)}*Math.pow(x,4) - {Beta(5)}*Math.pow(x,5) - {Beta(6)}*Math.pow(x,6) - {Beta(7)}*Math.pow(x,7) - {Beta(8)}*Math.pow(x,8)))"  # noqa
    js_parameters = [
        'Background', 'Beta(1)', 'Beta(2)', 'Beta(3)', 'Beta(4)',
        'Beta(5)', 'Beta(6)', 'Beta(7)', 'Beta(8)']
    version = 3.2
    date = '05/26/2010'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'background': DefaultParams.param_generator(name='Background'),
        'beta1': DefaultParams.param_generator(name='Beta1'),
        'beta2': DefaultParams.param_generator(name='Beta2'),
        'beta3': DefaultParams.param_generator(name='Beta3'),
        'beta4': DefaultParams.param_generator(name='Beta4'),
        'beta5': DefaultParams.param_generator(name='Beta5'),
        'beta6': DefaultParams.param_generator(name='Beta6'),
        'beta7': DefaultParams.param_generator(name='Beta7'),
        'beta8': DefaultParams.param_generator(name='Beta8'),
        'restrict_beta': DefaultParams.restrict(f=0, d=1, n='Restrict Beta'),
        'degree_poly': DefaultParams.degree_poly(),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.dich_bmr,
        'bmr_type': DefaultParams.dich_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
    }

    def as_dfile(self):
        degree_poly = self.values['degree_poly'][0]

        params = ['beta{}'.format(i) for i in range(1, degree_poly + 1)]
        params.insert(0, 'background')

        return '\n'.join([
            self._dfile_print_header_rows(),
            '{} {}'.format(self.dataset.doses_used, degree_poly),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'restrict_beta',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr', 'bmr_type', 'confidence_level'),
            self._dfile_print_parameters(*params),
            self.dataset.as_dfile(),
        ])


class Multistage_33(Multistage_32):
    bmds_version_dir = 'v240'
    version = 3.3
    date = '02/28/2013'
    defaults = deepcopy(Multistage_32.defaults)
    defaults['max_iterations']['d'] = 500


class MultistageCancer_19(DichotomousCancer):
    # todo: add check that degree poly must be <=8
    minimum_DG = 2
    model_name = 'Multistage-Cancer'
    bmds_version_dir = 'v231'
    exe = 'cancer'
    exe_plot = '10cancer'
    js_formula = "{Background} + (1. - {Background}) * (1. - Math.exp( -1. * {Beta(1)}*x - {Beta(2)}*Math.pow(x,2) - {Beta(3)}*Math.pow(x,3) - {Beta(4)}*Math.pow(x,4) - {Beta(5)}*Math.pow(x,5) - {Beta(6)}*Math.pow(x,6) - {Beta(7)}*Math.pow(x,7) - {Beta(8)}*Math.pow(x,8)))"  # noqa
    js_parameters = [
        'Background', 'Beta(1)', 'Beta(2)', 'Beta(3)', 'Beta(4)',
        'Beta(5)', 'Beta(6)', 'Beta(7)', 'Beta(8)']
    version = 1.9
    date = '05/26/2010'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'background': DefaultParams.param_generator(name='Background'),
        'beta1': DefaultParams.param_generator(name='Beta1'),
        'beta2': DefaultParams.param_generator(name='Beta2'),
        'beta3': DefaultParams.param_generator(name='Beta3'),
        'beta4': DefaultParams.param_generator(name='Beta4'),
        'beta5': DefaultParams.param_generator(name='Beta5'),
        'beta6': DefaultParams.param_generator(name='Beta6'),
        'beta7': DefaultParams.param_generator(name='Beta7'),
        'beta8': DefaultParams.param_generator(name='Beta8'),
        'restrict_beta': DefaultParams.restrict(f=1, d=1, n='Restrict Beta'),
        'degree_poly': DefaultParams.degree_poly(),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.dich_bmr,
        'bmr_type': DefaultParams.dich_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
    }

    def as_dfile(self):
        degree_poly = self.values['degree_poly'][0]

        params = ['beta{}'.format(i) for i in range(1, degree_poly + 1)]
        params.insert(0, 'background')

        return '\n'.join([
            self._dfile_print_header_rows(),
            '{} {}'.format(self.dataset.doses_used, degree_poly),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'restrict_beta',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr', 'bmr_type', 'confidence_level'),
            self._dfile_print_parameters(*params),
            self.dataset.as_dfile(),
        ])


class MultistageCancer_110(MultistageCancer_19):
    bmds_version_dir = 'v240'
    version = 1.10
    date = '02/28/2013'
    defaults = deepcopy(MultistageCancer_19.defaults)
    defaults['max_iterations']['d'] = 500


class Weibull_215(Dichotomous):
    minimum_DG = 3
    model_name = 'Weibull'
    bmds_version_dir = 'v231'
    exe = 'weibull'
    exe_plot = '10weibull'
    js_formula = "{Background} + (1-{Background}) * (1 - Math.exp( -1.*{Slope} * Math.pow(x,{Power}) ))"  # noqa
    js_parameters = ['Background', 'Slope', 'Power']
    version = 2.15
    date = '10/28/2009'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'log_transform': DefaultParams.log_transform(d=0),
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'background': DefaultParams.param_generator(f=1),
        'slope': DefaultParams.param_generator(name='Slope'),
        'power': DefaultParams.param_generator(name='Power'),
        'restrict_power': DefaultParams.restrict(f=0, d=1, n='Restrict Power'),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.dich_bmr,
        'bmr_type': DefaultParams.dich_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
    }

    def as_dfile(self):
        return '\n'.join([
            self._dfile_print_header_rows(),
            str(self.dataset.doses_used),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'restrict_power', 'bmd_calculation',
                'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr', 'bmr_type', 'confidence_level'),
            self._dfile_print_parameters(
                'background', 'slope', 'power'),
            self.dataset.as_dfile(),
        ])


class Weibull_216(Weibull_215):
    bmds_version_dir = 'v240'
    version = 2.16
    date = '02/28/2013'
    defaults = deepcopy(Weibull_215.defaults)
    defaults['max_iterations']['d'] = 500


class LogProbit_32(Dichotomous):
    minimum_DG = 3
    model_name = 'LogProbit'
    bmds_version_dir = 'v231'
    exe = 'probit'
    exe_plot = '10probit'
    js_formula = "{background} + (1-{background}) * Math.normalcdf(0,1,{intercept} + {slope}*Math.log(x))"  # noqa
    js_parameters = ['background', 'intercept', 'slope']
    version = 3.2
    date = '10/28/2009'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'log_transform': DefaultParams.log_transform(d=1),
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'background': DefaultParams.param_generator(f=1),
        'slope': DefaultParams.param_generator(name='Slope'),
        'intercept': DefaultParams.param_generator(name='Intercept'),
        'restrict_slope': DefaultParams.restrict(f=0, d=1, n='Restrict Slope'),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.dich_bmr,
        'bmr_type': DefaultParams.dich_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
    }

    def as_dfile(self):
        return '\n'.join([
            self._dfile_print_header_rows(),
            str(self.dataset.doses_used),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'log_transform', 'restrict_slope',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr', 'bmr_type', 'confidence_level'),
            self._dfile_print_parameters(
                'background', 'slope', 'intercept'),
            self.dataset.as_dfile(),
        ])


class LogProbit_33(LogProbit_32):
    bmds_version_dir = 'v240'
    version = 3.3
    date = '02/28/2013'
    defaults = deepcopy(LogProbit_32.defaults)
    defaults['max_iterations']['d'] = 500


class Probit_32(Dichotomous):
    minimum_DG = 2
    model_name = 'Probit'
    bmds_version_dir = 'v231'
    exe = 'probit'
    exe_plot = '10probit'
    js_formula = "Math.normalcdf(0,1,{intercept} + {slope}*x)"
    js_parameters = ['intercept', 'slope']
    version = 3.2
    date = '10/28/2009'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'log_transform': DefaultParams.log_transform(d=0),
        'restrict_slope': DefaultParams.restrict(f=1, d=0, n='Restrict Slope'),
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'background': DefaultParams.param_generator(f=1),
        'slope': DefaultParams.param_generator(name='Slope'),
        'intercept': DefaultParams.param_generator(name='Intercept'),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.dich_bmr,
        'bmr_type': DefaultParams.dich_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
    }

    def as_dfile(self):
        return '\n'.join([
            self._dfile_print_header_rows(),
            str(self.dataset.doses_used),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'log_transform', 'restrict_slope',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr', 'bmr_type', 'confidence_level'),
            self._dfile_print_parameters(
                'background', 'slope', 'intercept'),
            self.dataset.as_dfile(),
        ])


class Probit_33(Probit_32):
    bmds_version_dir = 'v240'
    version = 3.3
    date = '02/28/2013'
    defaults = deepcopy(Probit_32.defaults)
    defaults['max_iterations']['d'] = 500


class Gamma_215(Dichotomous):
    minimum_DG = 3
    model_name = 'Gamma'
    bmds_version_dir = 'v231'
    exe = 'gamma'
    exe_plot = '10gammhit'
    js_formula = "{Background} + (1 - {Background}) * Math.GammaCDF(x*{Slope},{Power})"  # noqa
    js_parameters = ['Background', 'Slope', 'Power']
    version = 2.15
    date = '10/28/2009'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'background': DefaultParams.param_generator(f=1),
        'slope': DefaultParams.param_generator(name='Slope'),
        'power': DefaultParams.param_generator(name='Power'),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'restrict_power': DefaultParams.restrict(f=0, d=1, n='Restrict Power'),
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.dich_bmr,
        'bmr_type': DefaultParams.dich_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
    }

    def as_dfile(self):
        return '\n'.join([
            self._dfile_print_header_rows(),
            str(self.dataset.doses_used),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'restrict_power',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr', 'bmr_type', 'confidence_level'),
            self._dfile_print_parameters(
                'background', 'slope', 'power'),
            self.dataset.as_dfile(),
        ])


class Gamma_216(Gamma_215):
    bmds_version_dir = 'v240'
    version = 2.16
    date = '02/28/2013'
    defaults = deepcopy(Gamma_215.defaults)
    defaults['max_iterations']['d'] = 500


class LogLogistic_213(Dichotomous):
    minimum_DG = 3
    model_name = 'LogLogistic'
    bmds_version_dir = 'v231'
    exe = 'logist'
    exe_plot = '10logist'
    js_formula = "{background} + (1-{background})/( 1 + Math.exp(-1.*{intercept}-1.*{slope}*Math.log(x) ) )"  # noqa
    js_parameters = ['background', 'intercept', 'slope']
    version = 2.13
    date = '10/28/2009'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'log_transform': DefaultParams.log_transform(d=1),
        'restrict_slope': DefaultParams.restrict(f=1, d=1),
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'background': DefaultParams.param_generator(f=1),
        'slope': DefaultParams.param_generator(name='Slope'),
        'intercept': DefaultParams.param_generator(name='Intercept'),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.dich_bmr,
        'bmr_type': DefaultParams.dich_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
    }

    def as_dfile(self):
        return '\n'.join([
            self._dfile_print_header_rows(),
            str(self.dataset.doses_used),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'log_transform', 'restrict_slope',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr', 'bmr_type', 'confidence_level'),
            self._dfile_print_parameters(
                'background', 'slope', 'intercept'),
            self.dataset.as_dfile(),
        ])


class LogLogistic_214(LogLogistic_213):
    bmds_version_dir = 'v240'
    version = 2.14
    date = '02/28/2013'
    defaults = deepcopy(LogLogistic_213.defaults)
    defaults['max_iterations']['d'] = 500


class Logistic_213(Dichotomous):
    minimum_DG = 2
    model_name = 'Logistic'
    bmds_version_dir = 'v231'
    exe = 'logist'
    exe_plot = '10logist'
    js_formula = "1/( 1 + Math.exp(-1*{intercept}-{slope}*x ))"
    js_parameters = ['intercept', 'slope']
    version = 2.13
    date = '10/28/2009'
    defaults = {
        'bmdl_curve_calculation': DefaultParams.bmdl_curve_calculation,
        'log_transform': DefaultParams.log_transform(d=0),
        'restrict_slope': DefaultParams.restrict(f=1, d=0),
        'append_or_overwrite': DefaultParams.append_or_overwrite,
        'smooth_option': DefaultParams.smooth_option,
        'max_iterations': DefaultParams.max_iterations,
        'relative_fn_conv': DefaultParams.relative_fn_conv,
        'parameter_conv': DefaultParams.parameter_conv,
        'background': DefaultParams.param_generator(f=1),
        'slope': DefaultParams.param_generator(name='Slope'),
        'intercept': DefaultParams.param_generator(name='Intercept'),
        'bmd_calculation': DefaultParams.bmd_calculation,
        'bmdl_curve_calc': DefaultParams.bmdl_curve_calc,
        'dose_drop': DefaultParams.dose_drop,
        'bmr': DefaultParams.dich_bmr,
        'bmr_type': DefaultParams.dich_bmr_type,
        'confidence_level': DefaultParams.confidence_level,
    }

    def as_dfile(self):
        return '\n'.join([
            self._dfile_print_header_rows(),
            str(self.dataset.doses_used),
            self._dfile_print_options(
                'max_iterations', 'relative_fn_conv', 'parameter_conv',
                'bmdl_curve_calculation', 'log_transform', 'restrict_slope',
                'bmd_calculation', 'append_or_overwrite', 'smooth_option'),
            self._dfile_print_options(
                'bmr', 'bmr_type', 'confidence_level'),
            self._dfile_print_parameters(
                'background', 'slope', 'intercept'),
            self.dataset.as_dfile(),
        ])


class Logistic_214(Logistic_213):
    bmds_version_dir = 'v240'
    version = 2.14
    date = '02/28/2013'
    defaults = deepcopy(Logistic_213.defaults)
    defaults['max_iterations']['d'] = 500
