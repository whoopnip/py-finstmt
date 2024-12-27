from enum import Enum

import pandas as pd
import pytest

from tests.fixtures.data.capiq import *  # noqa: F401, F403
from tests.fixtures.data.stockrow_cat import *  # noqa: F401, F403
from tests.fixtures.data.stockrow_mar import *  # noqa: F401, F403


class DataFrequency(str, Enum):
    ANNUAL = "annual"
    QUARTERLY = "quarterly"


freqs = list(DataFrequency)


class DataSource(str, Enum):
    CAPIQ = "capiq"
    STOCKROW_CAT = "stockrow_cat"
    STOCKROW_MAR = "stockrow_mar"


sources = list(DataSource)


@pytest.fixture(params=freqs, scope="session")
def data_frequency(request):
    return request.param


@pytest.fixture(params=sources, scope="session")
def data_source(request):
    return request.param


def _build(income_df: pd.DataFrame, balance_df: pd.DataFrame) -> FinancialStatements:
    income_stmt = StatementSeries.from_df(income_df)
    bs_stmt = StatementSeries.from_df(balance_df)
    stmts = FinancialStatements(income_stmt, bs_stmt)
    return stmts


@pytest.fixture(scope="session")
def statement(data_frequency: DataFrequency, data_source: DataSource):
    if data_frequency == DataFrequency.ANNUAL:
        if data_source == DataSource.CAPIQ:
            inc_df = annual_capiq_income_df()
            bs_df = annual_capiq_bs_df()
        elif data_source == DataSource.STOCKROW_CAT:
            inc_df = annual_stockrow_income_df_cat()
            bs_df = annual_stockrow_bs_df_cat()
        elif data_source == DataSource.STOCKROW_MAR:
            inc_df = annual_stockrow_income_df_mar()
            bs_df = annual_stockrow_bs_df_mar()
        else:
            raise NotImplementedError
    elif data_frequency == DataFrequency.QUARTERLY:
        if data_source == DataSource.CAPIQ:
            inc_df = quarterly_capiq_income_df()
            bs_df = quarterly_capiq_bs_df()
        elif data_source == DataSource.STOCKROW_CAT:
            inc_df = quarterly_stockrow_income_df_cat()
            bs_df = quarterly_stockrow_bs_df_cat()
        elif data_source == DataSource.STOCKROW_MAR:
            inc_df = quarterly_stockrow_income_df_mar()
            bs_df = quarterly_stockrow_bs_df_mar()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    income_stmt = StatementSeries.from_df(
        inc_df,
        INCOME_STATEMENT_CONFIG.display_name,
        INCOME_STATEMENT_CONFIG.items_config_list,
    )
    bs_stmt = StatementSeries.from_df(
        bs_df, BALANCE_SHEET_CONFIG.display_name, BALANCE_SHEET_CONFIG.items_config_list
    )

    # metrics statement is all calculated fields
    metrics_stmt = StatementSeries.from_df(
        inc_df[0:1],
        METRICS_STATEMENT_CONFIG.display_name,
        METRICS_STATEMENT_CONFIG.items_config_list,
    )

    # Fix for capiq annual date mismatch
    dates = income_stmt.dates
    stmts = FinancialStatements([income_stmt, bs_stmt[dates], metrics_stmt[dates]])
    return stmts
