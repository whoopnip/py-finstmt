import datetime
import re
from typing import Optional, Union

import numpy as np
import pandas as pd

COLUMN_NAME_PATTERN = re.compile(
    r"((Restated)|(Reclassified)|(12 months)|(3 months)|(Q\d)|(\n))*(?P<date>\w\w\w-\d\d-\d\d\d\d)"
)


def load_capiq_df(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Loads financial statements downloaded from Capital IQ into a DataFrame which can be passed into
    IncomeStatements or BalanceSheets
    """
    df = pd.read_excel(file_path, index_col=0, sheet_name=sheet_name, skiprows=14)

    # Rename columns, extracting date
    col_names = [_extract_date(col) for col in df.columns]
    df.columns = col_names

    # Drop non period ends such as LTM
    valid_col_names = [col for col in col_names if col is not None]
    df = df[valid_col_names]

    # Fill in - with mising
    df = df.replace("-", np.nan)

    return df


def _extract_date(
    column_name: Union[pd.Timestamp, datetime.datetime, str]
) -> Optional[Union[pd.Timestamp, datetime.datetime]]:
    """
    Extracts column date from Capital IQ columns.

    Returns None for LTM columns
    """
    if isinstance(column_name, (datetime.datetime, pd.Timestamp)):
        return column_name

    match = COLUMN_NAME_PATTERN.match(column_name)

    if not match:
        return None

    date_str = match.group("date")
    return pd.to_datetime(date_str)
