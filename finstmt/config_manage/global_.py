import datetime

import pandas as pd

from finstmt.config.statement_config import (
    BALANCE_SHEET_CONFIG,
    INCOME_STATEMENT_CONFIG,
)
from finstmt.config_manage.data import DataConfigManager
from finstmt.config_manage.statementseries import StatementSeriesConfigManager
from finstmt.config_manage.statements import StatementsConfigManager

inc_data_config_mgr = DataConfigManager(INCOME_STATEMENT_CONFIG.items_config_list)
inc_stmt_config_mgr = StatementSeriesConfigManager(
    {pd.Timestamp(datetime.datetime.today()): inc_data_config_mgr}
)
bs_data_config_mgr = DataConfigManager(BALANCE_SHEET_CONFIG.items_config_list)
bs_stmt_config_mgr = StatementSeriesConfigManager(
    {pd.Timestamp(datetime.datetime.today()): bs_data_config_mgr}
)
CONFIG = StatementsConfigManager(
    config_managers={
        "income_statements": inc_stmt_config_mgr,
        "balance_sheets": bs_stmt_config_mgr,
    }
)
