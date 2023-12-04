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


def test_calculate(data_path):
    df = pd.read_csv(data_path / "datasets/transforms/polyk.csv")

    # using no max day; derived from data
    _, res = polyk.calculate(
        doses=df.dose.tolist(),
        day=df.day.tolist(),
        has_tumor=df.has_tumor.tolist(),
        k=3,
    )
    assert df.day.max() == 734
    assert np.allclose(res.adj_proportion, [0.1414, 0.2836, 0.5700, 0.6643], atol=1e-4)

    # fixing max_day to a 2 yr cancer bioassay duration
    _, res = polyk.calculate(
        doses=df.dose.tolist(),
        day=df.day.tolist(),
        has_tumor=df.has_tumor.tolist(),
        k=3,
        max_day=730,
    )
    assert np.allclose(res.adj_proportion, [0.1404, 0.2821, 0.5673, 0.6616], atol=1e-4)


class TestAdjustment:
    def test_docx(self, data_path, rewrite_data_files):
        df = pd.read_csv(data_path / "datasets/transforms/polyk.csv")

        tool = polyk.Adjustment(
            doses=df.dose.tolist(),
            day=df.day.tolist(),
            has_tumor=df.has_tumor.tolist(),
            k=3,
        )

        # excel
        xlsx = tool.to_excel()

        # docx
        docx = tool.to_docx()
        if rewrite_data_files:
            (data_path / "bmds3-polyk.xlsx").write_bytes(xlsx.getvalue())
            docx.save(data_path / "bmds3-polyk.docx")
