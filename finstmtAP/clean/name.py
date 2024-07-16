import string
from typing import Any

import pandas as pd


def standardize_names_in_series_index(series: pd.Series):
    """
    Used internally to standardize names in DataFrames before looking up in name configs to match DataFrame
    data to data classes

    Note: inplace
    """
    series.index = [standardize_name_for_look_up(name) for name in series.index]


def standardize_name_for_look_up(name: Any) -> str:
    """
    Used internally to standardize names in DataFrames before looking up in name configs to match DataFrame
    data to data classes
    """
    if not isinstance(name, str):
        return name

    name = name.lower().strip()
    name = " ".join(name.split("_"))
    name = name.translate(
        str.maketrans("", "", string.punctuation)
    )  # remove punctuation
    name = " ".join(
        [part for part in name.split(" ") if part]
    )  # ensure there is only a single space between words
    return name
