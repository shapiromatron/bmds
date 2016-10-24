# -*- coding: utf-8 -*-
import math

from .. import constants


class Rule(object):

    def __init__(self, failure_bin, **kwargs):
        self.failure_bin = failure_bin
        self.enabled = kwargs.get('enabled', True)
        self.threshold = kwargs.get('threshold', float('nan'))
        self.name = kwargs.get('name', self.default_name)
        self.description = kwargs.get('name', self.default_description)
        self.kwargs = kwargs

    def __unicode__(self):
        enabled = u'✓' if self.enabled else u'✕'
        binmoji = constants.BINMOJI[self.failure_bin]
        threshold = '' if math.isnan(self.threshold) else ', threshold={}'.format(self.threshold)
        return u'{0} {1} [bin={2}{3}]'.format(enabled, self.name, binmoji, threshold)

    def check(self, dataset, output):
        if self.enabled:
            return self.apply_rule(dataset, output)

    def apply_rule(self, dataset, output):
        # return tuple of (bin, notes) associated with rule or None
        raise NotImplementedError('Abstract method.')

    def _is_valid_number(self, val):
        # Ensure number is an int or float, not equal to special case -999.
        return val is not None and \
            val != -999 and \
            (isinstance(val, int) or isinstance(val, float))


class ValueExistsRule(Rule):
    """Test fails if value is not a valid float."""

    field_name = None

    def apply_rule(self, dataset, output):
        val = output.get(self.field_name)
        if self._is_valid_number(val):
            return constants.BIN_NO_CHANGE, None
        else:
            msg = '{} (={}) is not a valid number'.format(self.name, val)
            return self.failure_bin, msg


class BmdExists(ValueExistsRule):
    default_name = 'BMD exists'
    default_description = 'A BMD value was successfully calculated.'
    field_name = 'BMD'


class BmdlExists(ValueExistsRule):
    default_name = 'BMDL exists'
    default_description = 'A BMDL value was successfully calculated.'
    field_name = 'BMDL'


class BmduExists(ValueExistsRule):
    default_name = 'BMDU exists'
    default_description = 'A BMDU value was successfully calculated.'
    field_name = 'BMDU'


class AicExists(ValueExistsRule):
    default_name = 'AIC exists'
    default_description = 'An AIC value was successfully calculated.'
    field_name = 'AIC'


class RoiExists(ValueExistsRule):
    default_name = 'Residual of interest exists'
    default_description = 'A residual of interest can be calculated.'
    field_name = 'residual_of_interest'


class ValueGreaterThanRule(Rule):
    # Test fails if value is less-than threshold.

    field_name = None

    def _assert_greater_than(self, output):
        val = output.get(self.field_name)
        threshold = self.threshold

        if not self._is_valid_number(val):
            return

        if val > threshold:
            return constants.BIN_NO_CHANGE, None
        else:
            msg = '{} (={}) is less-than than threshold value {}'.format(
                self.name, val, threshold)
            return self.failure_bin, msg


class VarianceFit(ValueGreaterThanRule):
    default_name = 'Variance fit'
    default_description = ('The variance model appropriately captures the variance '
                           'of the dataset being modeled (based on p-test #3).')
    field_name = 'p_value3'


class GlobalFit(ValueGreaterThanRule):
    default_name = 'GGOF'
    default_description = ('The mean model is appropriately modeling the dataset, '
                           'based on the global-goodness-of-fit.')
    field_name = 'p_value4'


class ValueLessThanRule(Rule):
    """Test fails if value is greater-than threshold."""

    def get_value(self, dataset, output):
        raise NotImplemented('Requires implementation')

    def apply_rule(self, dataset, output):
        val = self.get_value(dataset, output)
        threshold = self.threshold

        if not self._is_valid_number(val):
            return

        if val < threshold:
            return constants.BIN_NO_CHANGE, None
        else:
            msg = '{} (={}) is greater-than than threshold value {}'.format(
                self.name, val, threshold)
            return self.failure_bin, msg


class BmdBmdlRatio(ValueLessThanRule):
    default_name = 'BMD to BMDL ratio'
    default_description = ('The ratio between the BMD and BMDL values is large, '
                           'indicating a large spread of estimates in the range of interest.')

    def get_value(self, dataset, output):
        bmd = output.get('BMD')
        bmdl = output.get('BMDL')
        if self._is_valid_number(bmd) and self._is_valid_number(bmdl):
            return bmd / bmdl


class RoiFit(ValueLessThanRule):
    default_name = 'Residual of interest'
    default_description = ('The absolute value of the residual at the dose closest '
                           'to the BMD is large, which means the model may not be '
                           'estimating well in this range.')

    def get_value(self, dataset, output):
        return output.get('residual_of_interest')


class HighBmd(ValueLessThanRule):
    default_name = 'High BMD'
    default_description = ('The BMD estimate is greater than the maximum dose, '
                           'thus the model is extrapolating a result.')

    def get_value(self, dataset, output):
        max_dose = max(dataset.doses)
        bmd = output.get('BMD')
        if self._is_valid_number(max_dose) and self._is_valid_number(bmd):
            return bmd / max_dose


class HighBmdl(ValueLessThanRule):
    default_name = 'High BMDL'
    default_description = ('The BMDL estimate is greater than the maximum dose, '
                           'thus the model is extrapolating a result.')

    def get_value(self, dataset, output):
        max_dose = max(dataset.doses)
        bmdl = output.get('BMDL')
        if self._is_valid_number(max_dose) and self._is_valid_number(bmdl):
            return bmdl / max_dose


class LowBmd(ValueLessThanRule):
    default_name = 'Low BMD'
    default_description = 'The BMD estimate is lower than the lowest non-zero dose.'

    def get_value(self, dataset, output):
        min_dose = min([d for d in dataset.doses if d > 0])
        bmd = output.get('BMD')
        if self._is_valid_number(min_dose) and self._is_valid_number(bmd):
            return min_dose / bmd


class LowBmdl(ValueLessThanRule):
    default_name = 'Low BMDL'
    default_description = 'The BMDL estimate is lower than the lowest non-zero dose.'

    def get_value(self, dataset, output):
        min_dose = min([d for d in dataset.doses if d > 0])
        bmdl = output.get('BMDL')
        if self._is_valid_number(min_dose) and self._is_valid_number(bmdl):
            return min_dose / bmdl


class ControlResidual(ValueLessThanRule):
    default_name = 'Control residual'
    default_description = ('The absolute value of the residual at the control is large, '
                           'which is often used to estimate the BMR. Thus, the BMR estimate '
                           'may be inaccurate.')

    def get_value(self, dataset, output):
        if output.get('fit_residuals') and len(output['fit_residuals'] > 0):
            return abs(output['fit_residuals'][0])


class ControlStdevResiduals(ValueLessThanRule):
    default_name = 'Control stdev'
    default_description = ('The standard deviation estimate at the control is different '
                           'the the reported deviation, which is often used to estimate the BMR. '
                           'Thus, the BMR estimate may be inaccurate.')

    def get_value(self, dataset, output):
        if output.get('fit_est_stdev') and output.get('fit_stdev') and \
                len(output['fit_est_stdev'] > 0) and len(output['fit_stdev'] > 0):

            modeled = abs(output['fit_est_stdev'][0])
            actual = abs(output['fit_stdev'][0])

            if self._is_valid_number(modeled) and self._is_valid_number(actual):
                return abs(modeled / actual)


class CorrectVarianceModel(Rule):
    """Test fails if incorrect variance model is used for continuous datasets."""

    default_name = 'Variance type'
    default_description = 'The correct variance model was used (based on p-test #2).'

    def apply_rule(self, dataset, output):

        # TODO - get variance model, should be integer 0 or 1
        constant_variance = 1  # (or 0)

        p_value2 = output.get('p_value2')
        if p_value2 == '<0.0001':
            p_value2 = 0.0001

        if self._is_valid_number(p_value2):
            if (constant_variance == 1 and p_value2 >= 0.1) or \
                    (constant_variance == 0 and p_value2 <= 0.1):
                # correct variance model
                msg = None
            else:
                msg = 'Incorrect variance model (p-value 2 = ${})'.format(p_value2)
        else:
            msg = 'Correct variance model is undetermined (p-value 2 = ${})'.format(p_value2)

        if msg:
            return self.failure_bin, msg
        else:
            return constants.BIN_NO_CHANGE, None


class Warnings(Rule):
    """Test fails if any warnings exist."""

    default_name = 'Warnings'
    default_description = 'The model output file included additional warnings.'

    def apply_rule(self, dataset, output):
        warnings = output.get('warnings', [])
        if len(warnings) > 0:
            return constants.failure_bin, u'\n'.join(warnings)
