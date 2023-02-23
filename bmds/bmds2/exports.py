def df_ordered_dict(include_io=True) -> dict[str, list]:
    """
    Return an ordered defaultdict designed to create tabular exports from datasets
    """
    keys = [
        "dataset_index",
        "doses_dropped",
        "model_name",
        "model_index",
        "model_version",
        "has_output",
        "execution_halted",
        "BMD",
        "BMDL",
        "BMDU",
        "CSF",
        "AIC",
        "pvalue1",
        "pvalue2",
        "pvalue3",
        "pvalue4",
        "Chi2",
        "df",
        "residual_of_interest",
        "warnings",
        "logic_bin",
        "logic_cautions",
        "logic_warnings",
        "logic_failures",
        "recommended",
        "recommended_variable",
    ]

    if include_io:
        keys.extend(["dfile", "outfile", "stdout", "stderr"])

    return {key: [] for key in keys}
