import datetime

import pandas as pd

from finstmt.bs.config import BALANCE_SHEET_INPUT_ITEMS
from finstmt.config_manage.data import DataConfigManager
from finstmt.config_manage.statement import StatementConfigManager
from finstmt.config_manage.statements import StatementsConfigManager
from finstmt.inc.config import INCOME_STATEMENT_INPUT_ITEMS

inc_data_config_mgr = DataConfigManager(INCOME_STATEMENT_INPUT_ITEMS)
inc_stmt_config_mgr = StatementConfigManager(
    {pd.Timestamp(datetime.datetime.today()): inc_data_config_mgr}
)
bs_data_config_mgr = DataConfigManager(BALANCE_SHEET_INPUT_ITEMS)
bs_stmt_config_mgr = StatementConfigManager(
    {pd.Timestamp(datetime.datetime.today()): bs_data_config_mgr}
)
CONFIG = StatementsConfigManager(
    config_managers={
        "income_statements": inc_stmt_config_mgr,
        "balance_sheets": bs_stmt_config_mgr,
    }
)
