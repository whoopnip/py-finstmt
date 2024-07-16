from typing import Optional, Union

import pandas as pd


def add_cap_and_floor_to_df(
    df: pd.DataFrame,
    cap: Optional[Union[float, pd.Series]],
    floor: Optional[Union[float, pd.Series]],
):
    if cap is not None:
        df["cap"] = cap
    if floor is not None:
        df["floor"] = floor
