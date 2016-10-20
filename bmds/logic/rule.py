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

    def apply_rule(self, dtype, dataset, output):
        # return tuple of (bin, notes) associated with rule
        if self._is_enabled(dtype):
            method = getattr(self, self.rule_method_name)
            return method(dtype, dataset, output)

    def _is_enabled(self, dtype):
        return True

    def _is_valid_number(self, val):
        # Ensure number is an int or float, not equal to special case -999.
        return val is not None and \
            val != -999 and \
            (isinstance(val, int) or isinstance(val, float))

    def _assert_value_number(self, val):
        if self._is_valid_number(val):
            return constants.BIN_NO_CHANGE, None
        else:
            msg = '{} (={}) is not a valid number'.format(self.name, val)
            return self.failure_bin, msg

    def _assert_less_than(self, val, threshold=None):
        # Ensure value is less than threshold
        if threshold is None:
            threshold = self.threshold

        if val < threshold:
            return constants.BIN_NO_CHANGE, None
        else:
            msg = '{} (={}) is greater-than than threshold value {}'.format(
                self.name, val, threshold)
            return self.failure_bin, msg

    def _assert_greater_than(self, val, threshold=None):
        # Ensure value is greater than threshold
        if threshold is None:
            threshold = self.threshold

        if val > threshold:
            return constants.BIN_NO_CHANGE, None
        else:
            msg = '{} (={}) is less-than than threshold value {}'.format(
                self.name, val, threshold)
            return self.failure_bin, msg

    def _bmd_exists_rule(self, dtype, dataset, output):
        return self._assert_value_number(output.get('BMD'))

    def _bmdl_exists_rule(self, dtype, dataset, output):
        return self._assert_value_number(output.get('BMDL'))

    def _bmdu_exists_rule(self, dtype, dataset, output):
        return self._assert_value_number(output.get('BMDU'))

    def _aic_exists_rule(self, dtype, dataset, output):
        return self._assert_value_number(output.get('AIC'))

    def _residual_of_interest_exists_rule(self, dtype, dataset, output):
        # TODO - add
        return self._assert_value_number(output.get('residual_of_interest'))

    def _correct_variance_rule(self, dtype, dataset, output):
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

    def _variance_fit_rule(self, dtype, dataset, output):
        p_value3 = output.get('p_value3')
        if self._is_valid_number(p_value3):
            return self._assert_greater_than(p_value3)

    def _ggof_rule(self, dtype, dataset, output):
        p_value4 = output.get('p_value4')
        if self._is_valid_number(p_value4):
            return self._assert_greater_than(p_value4)

    def _bmd_bmdl_ratio_rule(self, dtype, dataset, output):
        bmd = output.get('BMD')
        bmdl = output.get('BMDL')
        if self._is_valid_number(bmd) and self._is_valid_number(bmdl):
            ratio = bmd / bmdl
            return self.assertLessThan(ratio)

    def _residual_of_interest_rule(self, dtype, dataset, output):
        roi = output.get('residual_of_interest')
        if self._is_valid_number(roi):
            return self._assert_less_than(roi)

    def _warnings_rule(self, dtype, dataset, output):
        warnings = output.get('warnings', [])
        if len(warnings) > 0:
            return constants.failure_bin, u'\n'.join(warnings)

    def _high_bmd_rule(self, dtype, dataset, output):
        max_dose = max(dataset.doses)
        bmd = output.get('BMD')
        if self._is_valid_number(max_dose) and self._is_valid_number(bmd):
            return self._assert_less_than(bmd / max_dose)

    def _high_bmdl_rule(self, dtype, dataset, output):
        max_dose = max(dataset.doses)
        bmdl = output.get('BMDL')
        if self._is_valid_number(max_dose) and self._is_valid_number(bmdl):
            return self._assert_less_than(bmdl / max_dose)

    def _low_bmd_rule(self, dtype, dataset, output):
        min_dose = min(dataset.doses)
        bmd = output.get('BMD')
        if self._is_valid_number(min_dose) and self._is_valid_number(bmd):
            return self._assert_less_than(min_dose / bmd)

    def _low_bmdl_rule(self, dtype, dataset, output):
        min_dose = min(dataset.doses)
        bmdl = output.get('BMDL')
        if self._is_valid_number(min_dose) and self._is_valid_number(bmdl):
            return self._assert_less_than(min_dose / bmdl)

    def _control_residual_rule(self, dtype, dataset, output):
        if output.get('fit_residuals') and len(output['fit_residuals'] > 0):
            resid = abs(output['fit_residuals'][0])
            if self._is_valid_number(resid):
                return self._assert_less_than(resid)

    def _control_stdev_rule(self, dtype, dataset, output):
        if output.get('fit_est_stdev') and output.get('fit_stdev') and \
                len(output['fit_est_stdev'] > 0) and len(output['fit_stdev'] > 0):

            modeled = abs(output['fit_est_stdev'][0])
            actual = abs(output['fit_stdev'][0])

            if self._is_valid_number(modeled) and self._is_valid_number(actual):
                ratio = abs(modeled / actual)
                return self._assert_less_than(ratio)
