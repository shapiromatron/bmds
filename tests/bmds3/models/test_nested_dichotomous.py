import json

import bmds
from bmds.bmds3.models import nested_dichotomous

# def test_bmds3_dichotomous_multistage(ddataset2):
#     # compare bmd, bmdl, bmdu, aic values
#     for degree, bmd_values, aic in [
#         (1, [17.680, 15.645, 20.062], 425.6),
#         (2, [48.016, 44.136, 51.240], 369.7),
#         (3, [63.873, 52.260, 72.126], 358.5),
#         (4, [63.871, 52.073, 72.725], 358.5),
#     ]:
#         settings = NestedDichotomousModelSettings(degree=degree)
#         model = nested_dichotomous.Multistage(ddataset2, settings)
#         result = model.execute()
#         actual = [result.bmd, result.bmdl, result.bmdu]
#         assert pytest.approx(bmd_values, abs=0.5) == actual
#         assert pytest.approx(aic, abs=5.0) == result.fit.aic


# def test_bmds3_dichotomous_session(ddataset2):
#     session = bmds.session.Bmds330(dataset=ddataset2)
#     session.add_default_models()
#     session.execute()
#     d = session.to_dict()
#     # ensure json-serializable
#     json.dumps(d)


# def test_bmds3_dichotomous_fit_parameters(ddataset2):
#     # check fit parameters for dichotomous modeling
#     model = nested_dichotomous.Logistic(ddataset2)
#     res = model.execute()
#     # overall fit
#     actual = [res.fit.loglikelihood, res.fit.aic, res.gof.p_value, res.gof.df, res.fit.chisq]
#     assert actual == pytest.approx([179.98, 363.96, 0.48, 3.0, 2.45], abs=0.01)
#     # scaled residuals
#     # TODO - this fix?
#     # assert res.gof.residual == pytest.approx([-1.08, -0.42, 0.94, -0.14, -0.46], abs=0.01)
#     # deviance
#     assert res.deviance.deviance == pytest.approx([-9999.0, 3.57, 307.681], abs=0.01)
#     assert res.deviance.df == pytest.approx([-9999, 3, 4])
#     assert res.deviance.p_value == pytest.approx([-9999.0, 0.311, 0.0], abs=0.01)


def test_bmds3_nested_dichotomous_placeholder(nd_dataset):
    some_analysis = nested_dichotomous.Logistic(nd_dataset)
    some_analysis.execute()

    assert some_analysis.structs.result.bmdsRes.BMD == 12.95166613

