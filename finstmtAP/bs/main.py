from dataclasses import dataclass
from typing import Dict

import pandas as pd

from finstmt.bs.config import BALANCE_SHEET_INPUT_ITEMS
from finstmt.findata.period_data import PeriodFinancialData
from finstmt.findata.statementsbase import FinStatementsBase


@dataclass
class BalanceSheets(FinStatementsBase):
    """
    Main class for holding balance sheet data. Usual way to construct is with the .from_df method.

    Examples:
        >>> bs_path = r'WMT Balance Sheet.xlsx'
        >>> bs_df = pd.read_excel(bs_path)
        >>> bs_data = BalanceSheets.from_df(bs_df)
    """

    statements: Dict[pd.Timestamp, PeriodFinancialData]  # type: ignore
    statement_name = "Balance Sheet"
    items_config_list = BALANCE_SHEET_INPUT_ITEMS
