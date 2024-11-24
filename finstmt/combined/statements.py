import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import pandas as pd
from sympy import Idx, IndexedBase, symbols
from typing_extensions import Self

from finstmt.check import item_series_is_empty
from finstmt.combined.combinator import (
    FinancialStatementsCombinator,
    StatementsCombinator,
)
from finstmt.config.statement_config import StatementConfig
from finstmt.config_manage.statements import StatementsConfigManager
from finstmt.exc import MismatchingDatesException
from finstmt.findata.statementsbase import FinStatementsBase
from finstmt.forecast.config import ForecastConfig
from finstmt.items.config import ItemConfig
from finstmt.logger import logger
from finstmt.resolver.solve import numpy_solve

if TYPE_CHECKING:
    from finstmt.forecast.statements import ForecastedFinancialStatements


@dataclass
class FinancialStatements:
    """
    Main class that holds all the financial statements.

    :param auto_adjust_config: Whether to automatically adjust the configuration based
        on the loaded data. Currently will turn forecasting off for items not in the data,
        and turn forecasting on for items normally calculated off those which are
        not in the data. For example, if gross_ppe is missing then will start forecasting
        net_ppe instead

    Examples:
        >>> bs_path = r'WMT Balance Sheet.xlsx'
        >>> inc_path = r'WMT Income Statement.xlsx'
        >>> bs_df = pd.read_excel(bs_path)
        >>> inc_df = pd.read_excel(inc_path)
        >>> bs_data = BalanceSheets.from_df(bs_df)
        >>> inc_data = IncomeStatements.from_df(inc_df)
        >>> stmts = FinancialStatements(inc_data, bs_data)
    """

    statements: List[FinStatementsBase]
    global_sympy_namespace: Dict[str, IndexedBase] = field(init=False, repr=False)
    calculate: bool = True
    auto_adjust_config: bool = True
    _combinator: StatementsCombinator[Self] = FinancialStatementsCombinator()  # type: ignore[assignment]

    def __post_init__(self):
        self.initialize_namespace()
        self.resolve_expressions()
        self.update_statements()
        self.resolve_statements()

    def initialize_namespace(self):
        t = symbols("t", cls=Idx)
        self.global_sympy_namespace = {"t": t}

        for stmt in self.statements:
            for config in stmt.items_config_list:
                expr = IndexedBase(config.key)
                self.global_sympy_namespace.update({config.key: expr})

    def resolve_expressions(self):
        eqns = []
        for statement in self.statements:
            eqns.extend(statement.get_expressions(self.global_sympy_namespace))

        all_to_solve = {}
        for eqn in eqns:
            expr = eqn.rhs - eqn.lhs
            all_to_solve[eqn.lhs] = expr

        to_solve_for = list(all_to_solve.keys())
        solve_exprs = list(all_to_solve.values())

        res = numpy_solve(solve_exprs, to_solve_for)

        for k, v in res.items():
            statement_item_key = k.base
            period_index = k.indices[0]
            statement_item_value = v
            for stmt in self.statements:
                stmt.update_statement_item_calculated_value(
                    statement_item_key, period_index, statement_item_value
                )

    def update_statements(self):
        for stmt in self.statements:
            stmt.df = stmt.to_df()

    def resolve_statements(self):
        from finstmt.resolver.history import StatementsResolver

        self._create_config_from_statements()

        if self.calculate:
            resolver = StatementsResolver(self)
            new_stmts = resolver.to_statements(auto_adjust_config=self.auto_adjust_config)
            self.statements = new_stmts.statements
            self._create_config_from_statements()

    def _create_config_from_statements(self):
        config_dict = {}
        for stmt_timeseries in self.statements:
            config_dict[stmt_timeseries.statement_name] = stmt_timeseries.config
        self.config = StatementsConfigManager(config_managers=config_dict)
        if self.auto_adjust_config:
            self._adjust_config_based_on_data()

    def _adjust_config_based_on_data(self):
        for item in self.config.items:
            if self.item_is_empty(item.key):
                if self.config.get(item.key).forecast_config.plug:
                    # It is OK for plug items to be empty, won't affect the forecast
                    continue

                # Useless to make forecasts on empty items
                logger.debug(f"Setting {item.key} to not forecast as it is empty")
                item.forecast_config.make_forecast = False
                # But this may mean another item should be forecasted instead.
                # E.g. normally net_ppe is calculated from gross_ppe and dep,
                # so it is not forecasted. But if gross_ppe is missing from
                # the data, then net_ppe should be forecasted directly.

                # So first, get the equations involving this item to determine
                # what other items are related to this one
                relevant_eqs = self.config.eqs_involving(item.key)
                relevant_keys: Set[str] = {item.key}
                for eq in relevant_eqs:
                    relevant_keys.add(self.config._expr_to_keys(eq.lhs)[0])
                    relevant_keys.update(set(self.config._expr_to_keys(eq.rhs)))
                relevant_keys.remove(item.key)
                for key in relevant_keys:
                    if self.item_is_empty(key):
                        continue
                    conf = self.config.get(key)
                    if conf.expr_str is None:
                        # Not a calculated item, so it doesn't make sense to turn forecasting on
                        continue

                    # Check to make sure that all components of the calculated item are also empty
                    expr = self.config.expr_for(key)
                    component_keys = self.config._expr_to_keys(expr)
                    all_component_items_are_empty = True
                    for c_key in component_keys:
                        if not self.item_is_empty(c_key):
                            all_component_items_are_empty = False
                    if not all_component_items_are_empty:
                        continue
                    # Now this is a calculated item which is non-empty, and all the components of the
                    # calculated are empty, so we need to forecast this item instead
                    logger.debug(
                        f"Setting {conf.key} to forecast as it is a calculated item which is not empty "
                        f"and yet none of the components have data"
                    )
                    conf.forecast_config.make_forecast = True

    def change(self, data_key: str) -> pd.Series:
        """
        Get the change between this period and last for a data series

        :param data_key: key of variable, how it would be accessed with FinancialStatements.data_key
        """
        series = getattr(self, data_key)
        return series - self.lag(data_key, 1)

    def lag(self, data_key: str, num_lags: int) -> pd.Series:
        """
        Get a data series lagged for a number of periods

        :param data_key: key of variable, how it would be accessed with FinancialStatements.data_key
        :param num_lags: Number of lags
        """
        series = getattr(self, data_key)
        return series.shift(num_lags)

    def item_is_empty(self, data_key: str) -> bool:
        """
        Whether the passed item has no data

        :param data_key: key of variable, how it would be accessed with FinancialStatements.data_key
        :return:
        """
        series = getattr(self, data_key)
        return item_series_is_empty(series)

    def _repr_html_(self):
        result = ""
        for stmt_timeseries in self.statements:
            result += f"""
            <h2>{stmt_timeseries.statement_name}</h2>
            {stmt_timeseries._repr_html_()}
            """
        return result

    def __getattr__(self, item):
        for stmt in self.statements:
            if item in dir(stmt):
                return getattr(stmt, item)

        raise AttributeError(item)

    # get a list of the hetrogeneous statements for a given date
    def __getitem__(self, item):
        stmts_hetrogeneous = []
        if not isinstance(item, (list, tuple)):
            date_item = pd.to_datetime(item)
            for stmt_timeseries in self.statements:
                stmts_hetrogeneous.append(
                    FinStatementsBase(
                        {date_item: stmt_timeseries[item]},
                        stmt_timeseries.items_config_list,
                        stmt_timeseries.statement_name,
                    )
                )
        else:
            for stmt in self.statements:
                stmts_hetrogeneous.append(stmt[item])

        return FinancialStatements(stmts_hetrogeneous, self.global_sympy_namespace)

    def __dir__(self):
        normal_attrs = [
            "forecast",
            "forecasts",
            "forecast_assumptions",
            "dates",
            "copy",
        ]
        all_config_items = []
        for stmt in self.statements:
            all_config_items.extend(stmt.config.items)

        item_attrs = [config_item.key for config_item in all_config_items]
        return normal_attrs + item_attrs

    def forecast(self, **kwargs) -> "ForecastedFinancialStatements":
        """
        Run a forecast, returning forecasted financial statements

        :param kwargs: Attributes of :class:`finstmt.forecast.config.ForecastConfig`

        :Examples:

            >>> stmts.forecast(periods=2)

        """
        from finstmt.resolver.forecast import ForecastResolver

        if "bs_diff_max" in kwargs:
            bs_diff_max = kwargs["bs_diff_max"]
        else:
            bs_diff_max = ForecastConfig.bs_diff_max

        if "balance" in kwargs:
            balance = kwargs["balance"]
        else:
            balance = ForecastConfig.balance

        if "timeout" in kwargs:
            timeout = kwargs["timeout"]
        else:
            timeout = ForecastConfig.timeout

        self._validate_dates()

        all_forecast_dict = {}
        all_results = {}
        for stmt in self.statements:
            forecast_dict, results = stmt._forecast(self, **kwargs)
            all_forecast_dict.update(forecast_dict)
            all_results.update(results)

        resolver = ForecastResolver(
            self, all_forecast_dict, all_results, bs_diff_max, timeout, balance=balance
        )

        obj = resolver.to_statements()
        return obj

    @property
    def forecast_assumptions(self) -> pd.DataFrame:
        all_series = []
        for config in self.all_config_items:
            if not config.forecast_config.make_forecast:
                continue
            config_series = config.forecast_config.to_series()
            config_series.name = config.display_name
            all_series.append(config_series)
        return pd.concat(all_series, axis=1).T

    @property
    def all_config_items(self) -> List[ItemConfig]:
        conf_items = []
        for stmts in self.statements:
            conf_items.extend(stmts.config.items)
        return conf_items

    @property
    def dates(self) -> List[pd.Timestamp]:
        self._validate_dates()
        return list(self.balance_sheets.statements.keys())

    def _validate_dates(self):
        for stmts1 in self.statements:
            for stmts2 in self.statements:
                stmts1_dates = set(stmts1.statements.keys())
                stmts2_dates = set(stmts2.statements.keys())
                if stmts1_dates != stmts2_dates:
                    stmts1_unique = stmts1_dates.difference(stmts2_dates)
                    stmts2_unique = stmts2_dates.difference(stmts1_dates)
                    message = "Got mismatching dates between historical statements. "
                    if stmts1_unique:
                        message += f"Balance sheet has {stmts1_unique} dates not in Income Statement. "
                    if stmts2_unique:
                        message += f"Income Statement has {stmts2_unique} dates not in Balance Sheet. "
                    raise MismatchingDatesException(message)

    def copy(self, **updates) -> Self:
        return dataclasses.replace(self, **updates)

    def __add__(self, other) -> Self:
        return self._combinator.add(self, other)

    def __radd__(self, other) -> Self:
        return self.__add__(other)

    def __sub__(self, other) -> Self:
        return self._combinator.subtract(self, other)

    def __rsub__(self, other) -> Self:
        return (-1 * self) + other

    def __mul__(self, other) -> Self:
        return self._combinator.multiply(self, other)

    def __rmul__(self, other) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other) -> Self:
        return self._combinator.divide(self, other)

    def __rtruediv__(self, other):
        # TODO [#41]: implement right division for statements
        raise NotImplementedError(
            f"cannot divide type {type(other)} by type {type(self)}"
        )

    def __round__(self, n: Optional[int] = None) -> Self:
        new_statements = self.copy()
        for stmt in new_statements.statements:
            stmt = round(stmt, n)  # type: ignore
        return new_statements

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        statement_config_list: List[StatementConfig],
        disp_unextracted: bool = True,
    ):
        """
        DataFrame must have columns as dates and index as names of financial statement items
        """
        dates = list(df.columns)
        dates.sort(key=lambda t: pd.to_datetime(t))

        stmts = []
        for statment_config in statement_config_list:
            stmts.append(
                FinStatementsBase.from_df(
                    df,
                    statment_config.display_name,
                    statment_config.items_config_list,
                    disp_unextracted=disp_unextracted,
                )
            )

        return cls(stmts)

    @classmethod
    def from_yaml_config(cls, df: pd.DataFrame, config_path: str, disp_unextracted: bool = True):
        """
        Create FinancialStatements from DataFrame using YAML config file
        
        :param df: DataFrame with financial data
        :param config_path: Path to YAML config file
        :param disp_unextracted: Whether to display unextracted items
        :return: FinancialStatements object
        """
        from finstmt.config.config_loader import load_yaml_config
        statement_configs = load_yaml_config(config_path)
        return cls.from_df(df, statement_configs, disp_unextracted)

    def to_excel(self, filepath: str, separate_sheets: bool = True) -> None:
        """
        Save the financial statements to an Excel file with statement headers.
        
        :param filepath: Path where the Excel file should be saved
        :param separate_sheets: If True, creates separate sheet for each statement. 
                              If False, combines all statements into one sheet
        """
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            workbook = writer.book
            money_fmt = workbook.add_format({
                'num_format': '$#,##0',
                'align': 'right'
            })
            header_fmt = workbook.add_format({
                'bold': True,
                'align': 'center'
            })
            title_fmt = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'align': 'left'
            })
            
            if separate_sheets:
                for stmt in self.statements:
                    df = stmt.df.copy()
                    df.fillna(0, inplace=True)
                    df.columns = [pd.to_datetime(col).strftime("%m/%d/%Y") for col in df.columns]
                    
                    sheet_name = stmt.statement_name
                    # Write statement name first, then data starting one row down
                    df.to_excel(writer, sheet_name=sheet_name, startrow=1)
                    
                    worksheet = writer.sheets[sheet_name]
                    worksheet.write(0, 0, sheet_name, title_fmt)
                    
                    for idx, col in enumerate(df.columns, start=1):
                        worksheet.set_column(idx, idx, 15, money_fmt)
                    
                    worksheet.set_row(1, None, header_fmt)  # Headers now on row 1 instead of 0
                    worksheet.set_column(0, 0, 30)
            else:
                all_dfs = []
                current_row = 0
                
                for stmt in self.statements:
                    df = stmt.df.copy()
                    df.fillna(0, inplace=True)
                    df.index = [f"{idx}" for idx in df.index]  # Don't need statement prefix in index anymore
                    all_dfs.append((stmt.statement_name, df))
                
                # Create single worksheet
                worksheet = workbook.add_worksheet('Financial Statements')
                
                # Write each statement with its header
                for stmt_name, df in all_dfs:
                    # Write statement header
                    worksheet.write(current_row, 0, stmt_name, title_fmt)
                    current_row += 1
                    
                    # Convert df to formatted dates
                    df.columns = [pd.to_datetime(col).strftime("%m/%d/%Y") for col in df.columns]
                    
                    # Write column headers
                    for idx, col in enumerate(df.columns):
                        worksheet.write(current_row, idx + 1, col, header_fmt)
                    
                    # Write index
                    for idx, row in enumerate(df.index):
                        worksheet.write(current_row + 1 + idx, 0, row)
                    
                    # Write data
                    for row_idx, row in enumerate(df.values):
                        for col_idx, value in enumerate(row):
                            worksheet.write(current_row + 1 + row_idx, col_idx + 1, value, money_fmt)
                    
                    current_row += len(df.index) + 2  # Move past data plus add a blank row
                
                # Set column widths
                worksheet.set_column(0, 0, 30)  # First column wider for labels
                worksheet.set_column(1, len(df.columns), 15)  # Data columns
