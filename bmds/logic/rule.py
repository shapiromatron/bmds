from .. import constants


class Rule(object):

    def __init__(self, name, description, failure_bin,
                 rule_method_name, threshold=None, enabled=True):
        self.name = name
        self.description = description
        self.failure_bin = failure_bin
        self.rule_method_name = rule_method_name
        self.threshold = threshold
        self.enabled = enabled

    def check(self, dtype, dataset, output):
        if self._is_enabled(dtype):
            return self.apply_rule(dtype, dataset, output)

    def apply_rule(self, dtype, dataset, output):
        # return tuple of (bin, notes) associated with rule or None
        raise NotImplementedError('Abstract method.')

    def _is_enabled(self, dtype):
        return True

    def _is_valid_number(self, val):
        # Ensure number is an int or float, not equal to special case -999.
        return val is not None and \
            val != -999 and \
            (isinstance(val, int) or isinstance(val, float))


class ValueExistsRule(Rule):
    # Test fails if value is not a valid float.

    FIELD_NAME = None

    def apply_rule(self, dtype, dataset, output):
        val = output.get(self.FIELD_NAME)
        if self._is_valid_number(val):
            return constants.BIN_NO_CHANGE, None
        else:
            msg = '{} (={}) is not a valid number'.format(self.name, val)
            return self.failure_bin, msg


class BmdExists(ValueExistsRule):
    FIELD_NAME = 'BMD'


class BmdlExists(ValueExistsRule):
    FIELD_NAME = 'BMDL'


class BmduExists(ValueExistsRule):
    FIELD_NAME = 'BMDU'


class AicExists(ValueExistsRule):
    FIELD_NAME = 'AIC'


class RoiExists(ValueExistsRule):
    FIELD_NAME = 'residual_of_interest'


class ValueGreaterThanRule(Rule):
    # Test fails if value is less-than threshold.

    FIELD_NAME = None

    def _assert_greater_than(self, output):
        val = output.get(self.FIELD_NAME)
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
    FIELD_NAME = 'p_value3'


class Fit(ValueGreaterThanRule):
    FIELD_NAME = 'p_value4'


class ValueLessThanRule(Rule):
    # Test fails if value is greater-than threshold.

    def get_value(self, dtype, dataset, output):
        raise NotImplemented('Requires implementation')

    def apply_rule(self, dtype, dataset, output):
        val = self.get_value(dtype, dataset, output)
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

    def get_value(self, dtype, dataset, output):
        bmd = output.get('BMD')
        bmdl = output.get('BMDL')
        if self._is_valid_number(bmd) and self._is_valid_number(bmdl):
            return bmd / bmdl


class RoiFit(ValueLessThanRule):

    def get_value(self, dtype, dataset, output):
        return output.get('residual_of_interest')


class HighBmd(ValueLessThanRule):

    def get_value(self, dtype, dataset, output):
        max_dose = max(dataset.doses)
        bmd = output.get('BMD')
        if self._is_valid_number(max_dose) and self._is_valid_number(bmd):
            return bmd / max_dose


class HighBmdl(ValueLessThanRule):

    def get_value(self, dtype, dataset, output):
        max_dose = max(dataset.doses)
        bmdl = output.get('BMDL')
        if self._is_valid_number(max_dose) and self._is_valid_number(bmdl):
            return bmdl / max_dose


class LowBmd(ValueLessThanRule):

    def get_value(self, dtype, dataset, output):
        min_dose = min(dataset.doses)
        bmd = output.get('BMD')
        if self._is_valid_number(min_dose) and self._is_valid_number(bmd):
            return min_dose / bmd


class LowBmdl(ValueLessThanRule):

    def get_value(self, dtype, dataset, output):
        min_dose = min(dataset.doses)
        bmdl = output.get('BMDL')
        if self._is_valid_number(min_dose) and self._is_valid_number(bmdl):
            return min_dose / bmdl


class ControlResiduals(ValueLessThanRule):

    def get_value(self, dtype, dataset, output):
        if output.get('fit_residuals') and len(output['fit_residuals'] > 0):
            return abs(output['fit_residuals'][0])


class ControlStdevResiduals(ValueLessThanRule):

    def get_value(self, dtype, dataset, output):
        if output.get('fit_est_stdev') and output.get('fit_stdev') and \
                len(output['fit_est_stdev'] > 0) and len(output['fit_stdev'] > 0):

            modeled = abs(output['fit_est_stdev'][0])
            actual = abs(output['fit_stdev'][0])

            if self._is_valid_number(modeled) and self._is_valid_number(actual):
                return abs(modeled / actual)


class CorrectVarianceModel(Rule):
    # Test fails if incorrect variance model is used for continuous datasets.

    def apply_rule(self, dtype, dataset, output):

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
    # Test fails if any warnings exist.

    def apply_rule(self, dtype, dataset, output):
        warnings = output.get('warnings', [])
        if len(warnings) > 0:
            return constants.failure_bin, u'\n'.join(warnings)
