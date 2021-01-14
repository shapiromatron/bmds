import pytest

import bmds


@pytest.mark.vcr()
def test_reporter(cdataset, ddataset, cidataset, rewrite_data_files):
    # Check that API works; if VIEW_REPORTS is in test environment then reports
    # are also written to disk for manual inspection.

    reporter1 = bmds.bmds2.Reporter()

    for ds in [cdataset, ddataset, cidataset]:
        ds.metadata.update(
            dataset_name="Smith 2017: Relative Liver Weight in Male SD Rats",
            dose_name="Dose (μg/m³)",
            response_name="Relative liver weight (mg/kg)",
            dose_units="μg/m³",
            response_units="mg/kg",
        )

    sessions = [
        bmds.BMDS.version("BMDS270", bmds.constants.DICHOTOMOUS, dataset=ddataset),
        bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS, dataset=cdataset),
        bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS, dataset=cidataset),
    ]

    for session in sessions:
        session.add_default_models()
        session.execute_and_recommend()
        reporter1.add_session(session)

    reporter2 = sessions[0].to_docx(all_models=True)

    if rewrite_data_files:
        reporter1.save("bmds_multi_session.docx")
        reporter2.save("bmds_single_session.docx")
