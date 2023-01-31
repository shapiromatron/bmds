import numpy as np
import pandas as pd

from bmds.datasets.transforms import polyk


def test_poly(data_path):
    df = pd.read_csv(data_path / "datasets/transforms/polyk.csv")
    assert df.shape == (200, 3)
    assert df.columns.values.tolist() == ["dose", "day", "has_tumor"]

    df2 = polyk.adjust_n(df)
    assert df2.shape == (200, 4)
    assert df2.columns.values.tolist() == ["dose", "day", "has_tumor", "adj_n"]

    df3 = polyk.summary_stats(df2)
    assert df3.shape == (4, 6)
    assert df3.columns.values.tolist() == [
        "dose",
        "n",
        "adj_n",
        "incidence",
        "proportion",
        "adj_proportion",
    ]
    assert np.allclose(df3.adj_proportion, [0.1414, 0.2836, 0.5700, 0.6643], atol=1e-4)
