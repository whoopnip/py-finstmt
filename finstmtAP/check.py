import pandas as pd


def item_series_is_empty(s: pd.Series) -> bool:
    if s.sum() != 0:
        return False
    for value in s:
        if value != 0:
            return False

    return True
