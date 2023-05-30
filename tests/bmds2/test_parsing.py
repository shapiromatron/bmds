from bmds import bmds2

expected_bad_anova_warnings = """THIS USUALLY MEANS THE MODEL HAS NOT CONVERGED!
BMDL computation failed.
Warning:  optimum may not have been found.  Bad completion code in Optimization routine.
Warning: Likelihood for fitted model larger than the Likelihood for model A3."""


def test_bad_anova_parsing(bad_anova_dataset, data_path):
    # check bad parsing of power
    with open(data_path / "outfiles/bad_anova_power.out") as f:
        outfile = f.read()
    model = bmds2.models.Power_219(bad_anova_dataset)
    output = model.parse_outfile(outfile)
    warnings = output["warnings"]
    actual = "\n".join(warnings)
    assert expected_bad_anova_warnings == actual

    # check separate issues with hill
    with open(data_path / "outfiles/bad_anova_hill.out") as f:
        outfile = f.read()
    model = bmds2.models.Hill_218(bad_anova_dataset)
    output = model.parse_outfile(outfile)
    assert output["Chi2"] == "1.#QNAN"
