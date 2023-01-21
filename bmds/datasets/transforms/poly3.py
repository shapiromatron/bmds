from typing import Optional

import pandas as pd


def adjust_incidence(
    df: pd.DataFrame, power: float = 3, max_day: Optional[int] = None
) -> pd.DataFrame:
    if df.columns.tolist() != ["dose", "day", "has_tumor"]:
        raise ValueError("Unexpected column names")
    df = df.copy()
    if max_day is None:
        max_day = df.day.max()
    df.loc[:, "adj_n"] = (df.query("has_tumor==0").day / max_day) ** power
    df.loc[:, "adj_n"] = df.loc[:, "adj_n"].fillna(1)
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.tolist() != ["dose", "day", "has_tumor", "adj_n"]:
        raise ValueError("Unexpected column names")
    grouped = df.groupby("dose")
    df2 = pd.DataFrame(
        data=[
            grouped.has_tumor.count().rename("n", inplace=True),
            grouped.adj_n.sum().rename("adj_n", inplace=True),
            grouped.has_tumor.sum().rename("incidence", inplace=True),
        ]
    ).T.reset_index()

    df2.loc[:, "proportion"] = df2.incidence / df2.n
    df2.loc[:, "adj_proportion"] = df2.incidence / df2.adj_n
    return df2
