import numpy as np
from sympy import nan
from finstmt.findata.statements import FinancialStatements
from finstmt.findata.statement_item import StatementItem
from finstmt.findata.statement_series import StatementSeries
from finstmt.findata.item_config import ItemConfig

def test_seed_value_is_none():
    item_config = ItemConfig(
        "cash_and_st_invest",
        "Cash and Short-Term Investments",
        expr_str="cash[t] + st_invest[t]",
    )
    statement_item = StatementItem(item_config)

    assert statement_item.seed_value is None
    assert statement_item.calculated_value is None
    assert statement_item.value is None


def test_update_statement_item_calculated_value_without_seed():
    item_config = ItemConfig(
        "cash_and_st_invest",
        "Cash and Short-Term Investments",
        expr_str="cash[t] + st_invest[t]",
    )
    statement_item = StatementItem(item_config)
    statement_item.update_statement_item_calculated_value(11)

    assert statement_item.seed_value is None
    assert statement_item.calculated_value == 11
    assert statement_item.value == 11


def test_update_statement_item_calculated_value_with_seed():
    item_config = ItemConfig(
        "cash_and_st_invest",
        "Cash and Short-Term Investments",
        expr_str="cash[t] + st_invest[t]",
    )
    statement_item = StatementItem(item_config, 10)
    statement_item.update_statement_item_calculated_value(11)

    assert statement_item.seed_value == 10
    assert statement_item.calculated_value == 11
    assert statement_item.value == 10


# These are more of an acceptance test
def test_default_value_for_non_calculated_field():
    config = [
        ItemConfig(
            key="cash",
            display_name="Cash",
            extract_names=["cash"],
        ),
        ItemConfig(
            key="investments",
            display_name="Investments",
            extract_names=["Investments"],
        ),
    ]
    dict = {
        "12/31/2022": {
            "cash": 1000
        }
    }

    stmt_timeseries = StatementSeries.from_dict(dict, "Test Statment", config)
    finstmts = FinancialStatements([stmt_timeseries])
    assert finstmts.investments[0] == 0


def test_calculated_field():
    config = [
        ItemConfig(
            key="cash",
            display_name="Cash",
            extract_names=["cash"],
        ),
        ItemConfig(
            key="equity",
            display_name="Equity",
            expr_str="cash[t]",
        ),
    ]
    dict = {
        "12/31/2022": {
            "cash": 1000,
        }
    }

    stmt_timeseries = StatementSeries.from_dict(dict, "Test Statment", config)
    finstmts = FinancialStatements([stmt_timeseries])
    assert finstmts.equity[0] == 1000


def test_nested_calculated_fields():
    config = [
        ItemConfig(
            key="cash",
            display_name="Cash",
            extract_names=["cash"],
        ),
        ItemConfig(
            key="assets",
            display_name="Assets",
            expr_str="cash[t]",
        ),
        ItemConfig(
            key="equity",
            display_name="Equity",
            expr_str="assets[t]",
        ),
    ]
    dict = {
        "12/31/2022": {
            "cash": 1000,
        }
    }

    stmt_timeseries = StatementSeries.from_dict(dict, "Test Statment", config)
    finstmts = FinancialStatements([stmt_timeseries])
    assert finstmts.equity[0] == 1000


# Equity is defined BEFORE Assets in config file where assets and equity are a calculated field
def test_nested_calculated_fields_out_of_order():
    config = [
        ItemConfig(
            key="cash",
            display_name="Cash",
            extract_names=["cash"],
        ),
        ItemConfig(
            key="equity",
            display_name="Equity",
            expr_str="assets[t]",
        ),
        ItemConfig(
            key="assets",
            display_name="Assets",
            expr_str="cash[t]",
        ),
    ]
    dict = {
        "12/31/2022": {
            "cash": 1000,
        }
    }

    stmt_timeseries = StatementSeries.from_dict(dict, "Test Statment", config)
    finstmts = FinancialStatements([stmt_timeseries])
    assert finstmts.equity[0] == 1000


def test_time_shift_calculated_field():
    config = [
        ItemConfig(
            key="cash",
            display_name="Cash",
            extract_names=["cash"],
        ),
        ItemConfig(
            key="cashDelta",
            display_name="Cash Delta",
            expr_str="cash[t] - cash[t-1]",
        ),
    ]
    dict = {
        "12/31/2021": {
            "cash": 1000,
        },
        "12/31/2022": {
            "cash": 1100,
        },
    }

    stmt_timeseries = StatementSeries.from_dict(dict, "Test Statment", config)
    finstmts = FinancialStatements([stmt_timeseries])
    print(type(finstmts.cashDelta[0]))
    assert np.isnan(finstmts.cashDelta[0])
    assert finstmts.cashDelta[1] == 100


