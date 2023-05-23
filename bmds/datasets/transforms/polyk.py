"""
Poly K adjustment test, original citation:

Bailer AJ, Portier CJ. Effects of treatment-induced mortality and tumor-induced mortality on tests
for carcinogenicity in small samples. Biometrics. 1988 Jun;44(2):417-31.
PMID: 3390507. DOI: 10.2307/2531856
"""

import pandas as pd


def adjust_n(df: pd.DataFrame, k: float | None = 3, max_day: int | None = None) -> pd.DataFrame:
    """Adjust the n for individual observations in a dataset.

    Args:
        df (pd.DataFrame): a DataFrame of dataset used to manipulate. Three columns:
            - dose (float >=0)
            - day (integer >=0)
            - has_tumor (integer, 0 or 1)
        k (Optional[float], optional, default 3): The adjustment term to apply
        max_day (Optional[int], optional): The maximum data. If specific, the value is used,
            otherwise, it is calculated from the maximum reported day in the dataset

    Returns:
        pd.DataFrame: A copy of the original dataframe, with an new column `adj_n`
    """
    columns = ["dose", "day", "has_tumor"]
    if df.columns.tolist() != columns:
        raise ValueError(f"Unexpected column names; expecting {columns}")
    if set(df.has_tumor.unique()) != {0, 1}:
        raise ValueError("Expected `has_tumor` values must be 0 and 1")
    df = df.copy()
    if max_day is None:
        max_day = df.day.max()
    df.loc[:, "adj_n"] = (df.query("has_tumor==0").day / max_day) ** k
    df.loc[:, "adj_n"] = df.loc[:, "adj_n"].fillna(1).clip(upper=1)
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics by group for poly k adjusted data, which can be used for
    dichotomous dose-response modeling.

    Args:
        df (pd.DataFrame): The input dataframe of individual response data with adjustment
            calculated from the `adjust_n` method above.

    Returns:
        pd.DataFrame: A dataframe of group-level values, both adjusted and unadjusted
    """
    columns = ["dose", "day", "has_tumor", "adj_n"]
    if df.columns.tolist() != columns:
        raise ValueError(f"Unexpected column names: {columns}")
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


def calculate(
    doses: list[float],
    day: list[int],
    has_tumor: list[int],
    k: float | None = 3,
    max_day: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate polyk adjustment on a dataset

    Args:
        doses (list[float]): A list of dose values
        day (list[int]): A list of days when observed
        has_tumor (list[int]): Binary flag for if entity had a tumor
        k (Optional[float], optional): Poly k adjustment value; defaults to 3.
        max_day (Optional[int], optional): Maximum observation day; defaults to calculating from
            dataset,

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Two data frames. The first is the individual data
            with the adjusted value, the second is summary data showing adjusted and unadjusted
            incidence data for use in dichotomous dose response modeling.
    """
    df = pd.DataFrame(dict(dose=doses, day=day, has_tumor=has_tumor))
    df2 = adjust_n(df, k, max_day)
    df3 = summary_stats(df2)
    return df2, df3
