# -*- coding: utf-8 -*-
import math

from .. import constants


class Rule(object):

    def __init__(self, failure_bin, **kwargs):
        self.failure_bin = failure_bin
        self.enabled = kwargs.get('enabled', True)
        self.threshold = kwargs.get('threshold', float('nan'))
        self.rule_name = kwargs.get('rule_name', self.default_rule_name)
        self.kwargs = kwargs

    def __unicode__(self):
        enabled = u'✓' if self.enabled else u'✕'
        binmoji = constants.BINMOJI[self.failure_bin]
        threshold = '' if math.isnan(self.threshold) \
            else ', threshold={}'.format(self.threshold)
        return u'{0} {1} [bin={2}{3}]'.format(
            enabled, self.rule_name, binmoji, threshold)

    def check(self, dataset, output):
        if self.enabled:
            return self.apply_rule(dataset, output)
        else:
            return self.return_pass()

    def return_pass(self):
        return constants.BIN_NO_CHANGE, None

    def apply_rule(self, dataset, output):
        # return tuple of (bin, notes) associated with rule or None
        raise NotImplementedError('Abstract method.')

    def get_failure_message(self, *args):
        raise NotImplementedError('Abstract method.')

    def _is_valid_number(self, val):
        # Ensure number is an int or float, not equal to special case -999.
        return val is not None and \
            val != -999 and \
            (isinstance(val, int) or isinstance(val, float))


class NumericValueExists(Rule):
    # Test succeeds if value is numeric and not -999
    field_name = None
    field_name_verbose = None

    def apply_rule(self, dataset, output):
        val = output.get(self.field_name)
        if self._is_valid_number(val):
            return self.return_pass()
        else:
            return self.failure_bin, self.get_failure_message()

    def get_failure_message(self):
        name = getattr(self, 'field_name_verbose')
        if name is None:
            name = self.field_name
        return '{} does not exist'.format(name)


class BmdExists(NumericValueExists):
    default_rule_name = 'BMD exists'
    field_name = 'BMD'


class BmdlExists(NumericValueExists):
    default_rule_name = 'BMDL exists'
    field_name = 'BMDL'


class BmduExists(NumericValueExists):
    default_rule_name = 'BMDU exists'
    field_name = 'BMDU'


class AicExists(NumericValueExists):
    default_rule_name = 'AIC exists'
    field_name = 'AIC'


class RoiExists(NumericValueExists):
    default_rule_name = 'Residual of interest exists'
    field_name = 'residual_of_interest'
    field_name_verbose = 'Residual of Interest'


class ShouldBeGreaterThan(Rule):
    # Test fails if value is less-than threshold.
    field_name = ''
    field_name_verbose = ''

    def apply_rule(self, dataset, output):
        val = output.get(self.field_name)
        threshold = self.threshold

        if not self._is_valid_number(val) or val >= threshold:
            return self.return_pass()
        else:
            return self.failure_bin, self.get_failure_message(val, threshold)

    def get_failure_message(self, val, threshold):
        name = self.field_name_verbose
        return '{} is less than threshold ({:.3} < {})'.format(name, float(val), threshold)


class VarianceFit(ShouldBeGreaterThan):
    default_rule_name = 'Variance fit'
    field_name = 'p_value3'
    field_name_verbose = 'Variance model fit p-value'


class GlobalFit(ShouldBeGreaterThan):
    default_rule_name = 'GGOF'
    field_name = 'p_value4'
    field_name_verbose = 'Goodness of fit p-value'


class ShouldBeLessThan(Rule):
    # Test fails if value is greater-than threshold.
    msg = ''  # w/ arguments for value and threshold

    def get_value(self, dataset, output):
        raise NotImplemented('Requires implementation')

    def apply_rule(self, dataset, output):
        val = self.get_value(dataset, output)
        threshold = self.threshold

        if not self._is_valid_number(val) or val <= threshold:
            return self.return_pass()
        else:
            return self.failure_bin, self.get_failure_message(val, threshold)

    def get_failure_message(self, val, threshold):
        name = self.field_name_verbose
        return '{} is greater than threshold ({:.3} > {})'.format(name, float(val), threshold)


class BmdBmdlRatio(ShouldBeLessThan):
    default_rule_name = 'BMD to BMDL ratio'
    field_name_verbose = 'BMD/BMDL ratio'

    def get_value(self, dataset, output):
        bmd = output.get('BMD')
        bmdl = output.get('BMDL')
        if self._is_valid_number(bmd) \
                and self._is_valid_number(bmdl) \
                and bmdl != 0:
            return bmd / bmdl


class RoiFit(ShouldBeLessThan):
    default_rule_name = 'Residual of interest'
    field_name_verbose = 'Residual of interest'

    def get_value(self, dataset, output):
        return output.get('residual_of_interest')


class HighBmd(ShouldBeLessThan):
    default_rule_name = 'High BMD'
    field_name_verbose = 'BMD/high dose ratio'

    def get_value(self, dataset, output):
        max_dose = max(dataset.doses)
        bmd = output.get('BMD')
        if self._is_valid_number(max_dose) \
                and self._is_valid_number(bmd)\
                and bmd != 0:
            return bmd / float(max_dose)


class HighBmdl(ShouldBeLessThan):
    default_rule_name = 'High BMDL'
    field_name_verbose = 'BMDL/high dose ratio'

    def get_value(self, dataset, output):
        max_dose = max(dataset.doses)
        bmdl = output.get('BMDL')
        if self._is_valid_number(max_dose) and \
                self._is_valid_number(bmdl) and \
                max_dose > 0:
            return bmdl / float(max_dose)


class LowBmd(ShouldBeLessThan):
    default_rule_name = 'Low BMD'
    field_name_verbose = 'BMD/minimum dose ratio'

    def get_value(self, dataset, output):
        min_dose = min([d for d in dataset.doses if d > 0])
        bmd = output.get('BMD')
        if self._is_valid_number(min_dose) and \
                self._is_valid_number(bmd) and \
                bmd > 0:
            return min_dose / float(bmd)


class LowBmdl(ShouldBeLessThan):
    default_rule_name = 'Low BMDL'
    field_name_verbose = 'BMDL/minimum dose ratio'

    def get_value(self, dataset, output):
        min_dose = min([d for d in dataset.doses if d > 0])
        bmdl = output.get('BMDL')
        if self._is_valid_number(min_dose) and \
                self._is_valid_number(bmdl) and \
                bmdl > 0:
            return min_dose / float(bmdl)


class ControlResidual(ShouldBeLessThan):
    default_rule_name = 'Control residual'
    field_name_verbose = 'Residual at lowest dose'

    def get_value(self, dataset, output):
        if output.get('fit_residuals') and len(output['fit_residuals']) > 0:
            try:
                return abs(output['fit_residuals'][0])
            except TypeError:
                return float('nan')


class ControlStdevResiduals(ShouldBeLessThan):
    default_rule_name = 'Control stdev'
    field_name_verbose = 'Ratio of modeled to actual stdev. at control'

    def get_value(self, dataset, output):
        if output.get('fit_est_stdev') and output.get('fit_stdev') and \
                len(output['fit_est_stdev']) > 0 and len(output['fit_stdev']) > 0:

            try:
                modeled = abs(output['fit_est_stdev'][0])
                actual = abs(output['fit_stdev'][0])
            except TypeError:
                return float('nan')

            if self._is_valid_number(modeled) and \
                    self._is_valid_number(actual) and \
                    modeled > 0 and actual > 0:
                return max(abs(modeled / actual), abs(actual / modeled))


class CorrectVarianceModel(Rule):
    # Check variance model (continuous datasets-only)
    default_rule_name = 'Variance type'

    def apply_rule(self, dataset, output):
        # 0 = non-homogeneous modeled variance => Var(i) = alpha*mean(i)^rho
        # 1 = constant variance => Var(i) = alpha*mean(i)
        if 'parameters' not in output:
            return self.return_pass()

        rho = output['parameters'].get('rho')
        constant_variance = 0 if rho else 1

        p_value2 = output.get('p_value2')
        if p_value2 == '<0.0001':
            p_value2 = 0.0001

        msg = None
        if self._is_valid_number(p_value2):
            if (constant_variance == 1 and p_value2 <= 0.1):
                msg = 'Incorrect variance model (p-value 2 = {}), constant variance selected'.format(p_value2)
            elif (constant_variance == 0 and p_value2 >= 0.1):
                msg = 'Incorrect variance model (p-value 2 = {}), modeled variance selected'.format(p_value2)
        else:
            msg = 'Correct variance model cannot be determined (p-value 2 = {})'.format(p_value2)

        if msg:
            return self.failure_bin, msg
        else:
            return self.return_pass()


class Warnings(Rule):
    # Test fails if any warnings exist.
    default_rule_name = 'Warnings'

    def get_failure_message(self, warnings):
        return u'Warning(s): {}'.format('; '.join(warnings))

    def apply_rule(self, dataset, output):
        warnings = output.get('warnings', [])
        if len(warnings) > 0:
            return self.failure_bin, self.get_failure_message(warnings)
        else:
            return self.return_pass()
