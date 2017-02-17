from math import log
from scipy import stats


class Test(object):

    def __init__(self):
        self.DF = -1
        self.CDF = -1
        self.SS = 0.0
        self.MSE = 0.0
        self.AIC = 0.0
        self.TEST = -1.0


class AnovaTests(object):

    @staticmethod
    def compute_likelihoods(n_obs, ns, ym, yd):
        Ntot = ns[0]
        Ntot = sum(ns)
        sigma2 = yd[0] * (ns[0] - 1)
        for i in range(1, n_obs):
            sigma2 += yd[i] * (ns[i] - 1)
        sigma2 = sigma2 / Ntot
        lkA1 = - Ntot * (1.0 + log(sigma2)) / 2.0
        lkA2 = - ns[0] * log(yd[0] * (ns[0] - 1) / ns[0]) / 2.0 - Ntot / 2.0
        for i in range(1, n_obs):
            lkA2 -= ns[i] * log(yd[i] * (ns[i] - 1) / ns[i]) / 2.0
        ybar = ym[0] * ns[0]
        for i in range(1, n_obs):
            ybar += ym[i] * ns[i]
        ybar = ybar / Ntot
        sigma2 = yd[0] * (ns[0] - 1) + ns[0] * (ym[0] - ybar) * (ym[0] - ybar)
        for i in range(1, n_obs):
            sigma2 += yd[i] * (ns[i] - 1) + \
                ns[i] * (ym[i] - ybar) * (ym[i] - ybar)
        sigma2 = sigma2 / Ntot
        lkR = -Ntot * (1.0 + log(sigma2)) / 2.0
        lkA3 = lkA1
        return lkA1, lkA2, lkA3, lkR

    @staticmethod
    def get_anova_c3_tests(nparm, n_obs, a1, a2, a3, ar):
        # Based on Hill model, modified from DTMS3ANOVAC.c

        # The xlk is not real
        xlk = (a1 + a2 + a3 + ar) / 4
        parm_known = 1
        anovaList = [Test() for i in range(5)]

        # Compute DF and assign LLK for each test
        anovaList[0].DF = n_obs + 1
        anovaList[0].SS = a1

        anovaList[1].DF = 2 * n_obs
        anovaList[1].SS = a2

        anovaList[2].DF = n_obs + 2 - parm_known
        anovaList[2].SS = a3

        anovaList[3].DF = 2
        anovaList[3].SS = ar

        anovaList[4].DF = nparm - 2
        anovaList[4].SS = xlk

        # Compute likelihood ratio MSE and CDF
        anovaList[0].MSE = 2 * (a2 - a1)
        anovaList[0].CDF = anovaList[1].DF - anovaList[0].DF

        anovaList[1].MSE = 2 * (a2 - a3)
        anovaList[1].CDF = anovaList[1].DF - anovaList[2].DF

        anovaList[2].MSE = 2 * (a3 - xlk)
        anovaList[2].CDF = anovaList[2].DF - anovaList[4].DF

        anovaList[3].MSE = 2 * (a2 - ar)
        anovaList[3].CDF = anovaList[1].DF - anovaList[3].DF

        for anova in anovaList:
            anova.AIC = -2 * (anova.SS - anova.DF)
            if (anova.MSE >= 0.0 and anova.CDF > 0):
                anova.TEST = 1 - stats.chi2.cdf(anova.MSE, anova.CDF)

        # Only return test 1, test 2 and test 3 in the order
        return (anovaList[3], anovaList[0], anovaList[1])

    @staticmethod
    def output_3tests(tests):
        if tests is None:
            return 'ANOVA cannot be calculated for this dataset.'

        outputs = [
            '                     Tests of Interest    ',
            '   Test    -2*log(Likelihood Ratio)  Test df        p-value    ',
        ]
        for i, test in enumerate(tests):
            outputs.append('   Test %d %20.6g %10d %16.4g' % (
                i + 1, test.MSE, test.CDF, test.TEST))
        return '\n'.join(outputs)
