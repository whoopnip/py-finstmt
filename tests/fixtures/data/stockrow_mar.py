import os

import pandas as pd
import pytest

from finstmt import FinancialStatements
from finstmt.config.statement_config import (
    BALANCE_SHEET_CONFIG,
    INCOME_STATEMENT_CONFIG,
)
from finstmt.findata.statement_series import StatementSeries
from tests.fixtures.data.common import DATA_PATH

STOCKROW_PATH = os.path.join(DATA_PATH, "stockrow")
MAR_PATH = os.path.join(STOCKROW_PATH, "MAR")


def annual_stockrow_income_df_mar() -> pd.DataFrame:
    annual_path = os.path.join(MAR_PATH, "annual_income.csv")
    df = pd.read_csv(annual_path, index_col=0)
    return df


@pytest.fixture
def annual_stockrow_income_stmt_mar() -> StatementSeries:
    stmt = StatementSeries.from_df(
        annual_stockrow_income_df_mar(),
        INCOME_STATEMENT_CONFIG.display_name,
        INCOME_STATEMENT_CONFIG.items_config_list,
    )
    return stmt


def annual_stockrow_bs_df_mar() -> pd.DataFrame:
    annual_path = os.path.join(MAR_PATH, "annual_bs.csv")
    df = pd.read_csv(annual_path, index_col=0)
    return df


@pytest.fixture
def annual_stockrow_bs_stmt_mar() -> StatementSeries:
    stmt = StatementSeries.from_df(
        annual_stockrow_bs_df_mar(),
        BALANCE_SHEET_CONFIG.display_name,
        BALANCE_SHEET_CONFIG.items_config_list,
    )
    return stmt


@pytest.fixture
def annual_stockrow_stmts_mar(
    annual_stockrow_income_stmt_mar, annual_stockrow_bs_stmt_mar
) -> FinancialStatements:
    stmts = FinancialStatements(
        [annual_stockrow_income_stmt_mar, annual_stockrow_bs_stmt_mar]
    )
    return stmts


def quarterly_stockrow_income_df_mar() -> pd.DataFrame:
    quarterly_path = os.path.join(MAR_PATH, "quarterly_income.csv")
    df = pd.read_csv(quarterly_path, index_col=0)
    return df


@pytest.fixture
def quarterly_stockrow_income_stmt_mar() -> StatementSeries:
    stmt = StatementSeries.from_df(
        quarterly_stockrow_income_df_mar(),
        BALANCE_SHEET_CONFIG.display_name,
        BALANCE_SHEET_CONFIG.items_config_list,
    )
    return stmt


def quarterly_stockrow_bs_df_mar() -> pd.DataFrame:
    quarterly_path = os.path.join(MAR_PATH, "quarterly_bs.csv")
    df = pd.read_csv(quarterly_path, index_col=0)
    return df


@pytest.fixture
def quarterly_stockrow_bs_stmt_mar() -> StatementSeries:
    stmt = StatementSeries.from_df(
        quarterly_stockrow_bs_df_mar(),
        INCOME_STATEMENT_CONFIG.display_name,
        INCOME_STATEMENT_CONFIG.items_config_list,
    )
    return stmt


@pytest.fixture
def quarterly_stockrow_stmts_mar(
    quarterly_stockrow_income_stmt_mar, quarterly_stockrow_bs_stmt_mar
) -> FinancialStatements:
    stmts = FinancialStatements(
        [quarterly_stockrow_income_stmt_mar, quarterly_stockrow_bs_stmt_mar]
    )
    return stmts
