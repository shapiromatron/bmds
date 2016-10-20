import re

from . import constants

EXPONENTIAL = 'E'
CTYPES = [constants.CONTINUOUS, EXPONENTIAL]


class OutputParser(object):

    # regex for finding numeric values
    re_num = r'[/+/-]?[0-9]+[/.]*[0-9]*[Ee+-]*[0-9]*'

    # line-skips by dataset type
    NUM_LINE_SKIPS_PARAMS = {
        constants.CONTINUOUS: 1,
        constants.DICHOTOMOUS: 1,
        constants.DICHOTOMOUS_CANCER: 1,
        EXPONENTIAL: 4,
    }

    NUM_LINE_SKIPS_FIT = {
        constants.CONTINUOUS: 3,
        constants.DICHOTOMOUS: 2,
        constants.DICHOTOMOUS_CANCER: 2,
    }

    NUM_LINE_SKIPS_AIC = {
        constants.CONTINUOUS: 1,
        EXPONENTIAL: 2,
    }

    def __init__(self, output_text, dtype, model_name):
        assert dtype in constants.DTYPES
        if 'exponential' in model_name.lower():
            dtype = EXPONENTIAL

        self.dtype = dtype
        self.output_text = output_text
        self.model_name = model_name
        self.parse_output_file()

    def parse_output_file(self):
        self.output = {}

        self._set_default_values()
        self._import_single_searches()
        self._import_warnings()
        self._import_line_by_line()
        self._calc_residual_of_interest()
        self._cleanup_nans()

    def _set_default_values(self):
        # set name
        self.output['model_name'] = self.model_name

        # initial setup
        vals = ('Chi2', 'df', 'p_value4', 'residual_of_interest')
        for val in vals:
            self.output[val] = -999

    def _import_line_by_line(self):
        if self.dtype in constants.DICH_DTYPES:
            fit_tbl = (
                'fit_dose',
                'fit_est_prob',
                'fit_estimated',
                'fit_observed',
                'fit_size',
                'fit_residuals',
            )
        elif self.dtype in CTYPES:
            fit_tbl = (
                'fit_dose',
                'fit_size',
                'fit_observed',
                'fit_estimated',
                'fit_stdev',
                'fit_est_stdev',
                'fit_residuals'
            )

        # set empty tables
        for val in fit_tbl:
            self.output[val] = []

        # begin line-by-line
        outs = self.output_text.splitlines()
        if self.dtype in constants.DICH_DTYPES:
            self._import_dich_vals()
            for i in range(len(outs)):
                if outs[i] == r'       Variable         Estimate        Std. Err.     Lower Conf. Limit   Upper Conf. Limit':  # noqa
                    self._lbl_parameter(outs, i)
                elif outs[i] == r'     Dose     Est._Prob.    Expected    Observed     Size       Residual':  # noqa
                    self._lbl_fit_cont_dich(outs, i, fit_tbl)

        elif self.dtype == constants.CONTINUOUS:
            for i in range(len(outs)):
                if outs[i] == r'       Variable         Estimate        Std. Err.     Lower Conf. Limit   Upper Conf. Limit':  # noqa
                    self._lbl_parameter(outs, i)
                elif outs[i] == r' Dose       N    Obs Mean     Est Mean   Obs Std Dev  Est Std Dev   Scaled Res.':  # noqa
                    self._lbl_fit_cont_dich(outs, i, fit_tbl)
                elif outs[i] == r'   Test    -2*log(Likelihood Ratio)  Test df        p-value    ':  # noqa
                    self._lbl_pvalue(outs, i)
                elif outs[i] == r"            Model      Log(likelihood)   # Param's      AIC":  # noqa
                    self._lbl_aic_cont_exp(outs, i)

        elif self.dtype == EXPONENTIAL:
            for i in range(len(outs)):
                if outs[i] == r'                     Parameter Estimates':
                    self._lbl_parameter(outs, i)
                elif outs[i] == r'     Dose      N         Obs Mean     Obs Std Dev':  # noqa
                    self._lbl_fit_exp(outs, i, 'observed')
                elif outs[i] == r'      Dose      Est Mean      Est Std     Scaled Residual':  # noqa
                    self._lbl_fit_exp(outs, i, 'estimated')
                elif outs[i] == r'     Test          -2*log(Likelihood Ratio)       D. F.         p-value':  # noqa
                    self._lbl_pvalue(outs, i)
                elif outs[i] == r"                     Model      Log(likelihood)      DF         AIC":  # noqa
                    self._lbl_aic_cont_exp(outs, i)

    def _cleanup_nans(self):
        # standardize possible errors
        fields = ('AIC', 'p_value1', 'p_value2', 'p_value3', 'p_value4')
        for field in fields:
            if field in self.output and self.output[field] in ['NA', 'N/A']:
                self.output[field] = -999

    def _calc_residual_of_interest(self):
        bmd = self.output['BMD']
        if bmd > 0 and len(self.output['fit_dose']) > 0:
            diff = abs(self.output['fit_dose'][0] - bmd)
            r = self.output['fit_residuals'][0]
            for j, val in enumerate(self.output['fit_dose']):
                if abs(val - bmd) < diff:
                    diff = abs(val - bmd)
                    r = self.output['fit_residuals'][j]
            self.output['residual_of_interest'] = r

    def _import_single_searches(self):
        """
        Look for simple one-line regex searches common across dataset types.

        If failed, then return -999.

        AIC is only handled in this method for dichotomous continuous and
        exponential and handled via separate methods.
        """
        searches = {
            # (?<!Setting ) is a special case for preventing
            # "Setting BMD = 100*(maximum dose)" matches
            'BMD': r'(?<!Setting )BMD = +(%s)' % self.re_num,
            'BMDL': r'BMDL = +(%s)' % self.re_num,
            'BMDU': r'BMDU = +(%s)' % self.re_num,
            'CSF': r'Multistage Cancer Slope Factor = +(%s)' % self.re_num,
            'AIC': r'AIC: +(%s)' % (self.re_num),
            'model_version': r'Version: ([\d\.]+);',
            'model_date': r'Date: ([\d/]+)',
        }
        for search in searches:
            m = re.search(searches[search], self.output_text)
            if m:
                try:
                    self.output[search] = float(m.group(1))
                except:
                    self.output[search] = m.group(1)
            else:
                self.output[search] = -999

    def _import_warnings(self):
        """
        Add custom warnings found in output files.

        Warnings in output files are searched for using this method; if a
        warning is found then it will be appended to the warnings list.
        """
        warnings = (
            r'Warning: BMDL computation is at best imprecise for these data',
            r'THE MODEL HAS PROBABLY NOT CONVERGED!!!',
            r'BMR value is not in the range of the mean function',
            r'BMD = 100\*\(maximum dose\)',
            r'BMDL computation failed\.',
        )
        self.output['warnings'] = []
        for warning in warnings:
            m = re.search(warning, self.output_text)
            if m:
                self.output['warnings'].append(m.group())

    def _import_dich_vals(self):
        """
        Import simple dichotomous values.

        Dichotomous values are found on one line, therefore one regex will
        return up to three possible matches for the Chi^2, degrees of freedom,
        and p-value.
        """
        m = re.search(r'Chi\^2 = ({0}|\w+) +d.f. = +({0}|\w+) +P-value = +({0}|\w+)'.format(self.re_num),  # noqa
                      self.output_text)
        cw = {1: "Chi2", 2: 'df', 3: 'p_value4'}
        for val in cw:
            try:
                self.output[cw[val]] = float(m.group(val))
            except:
                self.output[cw[val]] = -999

    def _lbl_parameter(self, outs, i):
        self.output['parameters'] = {}
        cw = {
            1: 'estimate',
            2: 'stdev',
            3: '95_low_limit',
            4: '95_high_limit',
        }
        i += self.NUM_LINE_SKIPS_PARAMS[self.dtype]
        while (len(outs[i].split()) > 0):
            vals = outs[i].split()
            self.output['parameters'][vals[0]] = {}
            for j in range(1, len(vals)):
                try:
                    self.output['parameters'][vals[0]][cw[j]] = float(vals[j])
                except:
                    self.output['parameters'][vals[0]][cw[j]] = vals[j]
            i += 1

    def _lbl_fit_cont_dich(self, outs, i, fit_tbl):
        # Line-by-line: find "Goodness  of  Fit" table
        i += self.NUM_LINE_SKIPS_FIT[self.dtype]
        while len(outs[i]) > 1:
            vals = outs[i].split()
            for j in range(len(vals)):
                try:
                    self.output[fit_tbl[j]].append(float(vals[j]))
                except:
                    self.output[fit_tbl[j]].append(vals[j])
            i += 1

    def _lbl_fit_exp(self, outs, i, table_name):
        # Line-by-line: find "Goodness  of  Fit" table - exponential - observed
        i += 2   # next line and dotted lines

        if table_name == 'observed':
            tbl_names = ('fit_dose', 'fit_size', 'fit_observed', 'fit_stdev')
            rng = range(len(outs[i].split()))
        elif table_name == 'estimated':
            tbl_names = ('fit_dose', 'fit_estimated', 'fit_est_stdev', 'fit_residuals')  # noqa
            rng = range(1, len(outs[i].split()))

        while len(outs[i]) > 0:
            vals = outs[i].split()
            for j in rng:  # skip dose, picked up w/ estimated
                try:
                    self.output[tbl_names[j]].append(float(vals[j]))
                except:
                    self.output[tbl_names[j]].apend(vals[j])
            i += 1

    def _lbl_pvalue(self, outs, i):
        # Line-by-line: find p-values (continuous)
        i += 2   # next line and blank line
        while len(outs[i]) > 0:
            vals = outs[i].split()
            if vals[1] in ['5a', '6a', '7a']:  # fix exponentials
                pvalue = '4'
            else:
                pvalue = vals[1]

            if vals[4] == '<':
                self.output['p_value' + pvalue] = '<0.0001'
            else:
                try:
                    self.output['p_value' + pvalue] = float(vals[4])
                except:
                    if vals[4] == '<.0001':
                        self.output['p_value' + pvalue] = '<0.0001'
                    else:
                        self.output['p_value' + pvalue] = vals[4]

            if pvalue == '4':
                self.output['Chi2'] = float(vals[2])
                self.output['df'] = float(vals[3])
            i += 1

    def _lbl_aic_cont_exp(self, outs, i):
        # Line-by-line: find AIC (continuous)
        i += self.NUM_LINE_SKIPS_AIC[self.dtype]
        field = ('fitted', '2', '3', '4', '5')  # continuous is "fitted"
        while len(outs[i]) > 0:
            vals = outs[i].split()
            if vals[0] in field:
                try:
                    self.output['AIC'] = float(vals[3])
                except:
                    self.output['AIC'] = vals[3]
            i += 1
