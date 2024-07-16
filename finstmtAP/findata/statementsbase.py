import operator
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from finstmt.check import item_series_is_empty
from finstmt.config_manage.data import DataConfigManager, _key_pct_of_key
from finstmt.config_manage.statement import StatementConfigManager
from finstmt.exc import (
    CouldNotParseException,
    MixedFrequencyException,
    NoSuchItemException,
)
from finstmt.findata.period_data import PeriodFinancialData
from finstmt.forecast.config import ForecastConfig
from finstmt.forecast.main import Forecast
from finstmt.items.config import ItemConfig
from finstmt.logger import logger


class FinStatementsBase:
    statements: Dict[pd.Timestamp, PeriodFinancialData]
    statement_name: str = "Base"
    items_config_list: List[ItemConfig]

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __post_init__(self):
        self.df = self.to_df()

        # Hook up prior statements to statements
        dates = list(self.statements.keys())
        dates.sort()
        prior_date = None
        for i, date in enumerate(dates):
            if i != 0:
                self.statements[date].prior_statement = self.statements[prior_date]
            prior_date = date

        # Create dictionary of individual time period configs to construct the entire statement config
        configs_dict = {}
        for date, statement in self.statements.items():
            configs_dict[date] = statement.config_manager
        self.config = StatementConfigManager(configs_dict)

    def _repr_html_(self):
        return self._formatted_df._repr_html_()

    # Get longitudenal series for a statement item
    def __getattr__(self, item):
        data_dict = {}
        for (
            date,
            statement,
        ) in self.statements.items():
            try:
                statement_value = getattr(statement, item)
            except AttributeError:
                # Should hit here on the first loop if this is an invalid item
                # Raise attribute error like normal.
                raise AttributeError(item)
            if pd.isnull(statement_value):
                statement_value = 0
            data_dict[date] = statement_value
        item_config: Optional[ItemConfig] = None
        try:
            item_config = self.config.get(item)
        except NoSuchItemException:
            pass
        return pd.Series(
            data_dict, name=item_config.display_name if item_config else item
        )

    def __getitem__(self, item):
        if not isinstance(item, (list, tuple)):
            date_item = pd.to_datetime(item)
            return self.statements[date_item]

        # Got multiple dates
        all_series = []
        for date_str in item:
            series = self.df[date_str]
            date = pd.to_datetime(date_str)
            series.name = date
            all_series.append(series)
        df = pd.concat(all_series, axis=1)

        return self.from_df(df, disp_unextracted=False)

    def __dir__(self):
        normal_attrs = [
            "statements",
            "to_df",
            "freq",
        ]
        item_attrs = dir(list(self.statements.values())[0])
        return normal_attrs + item_attrs

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        items_config_list: Optional[List[ItemConfig]] = None,
        disp_unextracted: bool = True,
    ):
        """
        DataFrame must have columns as dates and index as names of financial statement items
        """
        statements_dict = {}
        dates = list(df.columns)
        dates.sort(key=lambda t: pd.to_datetime(t))

        if items_config_list is None:
            config_manager = DataConfigManager(cls.items_config_list.copy())
        else:
            config_manager = DataConfigManager(items_config_list.copy())

        for col in dates:
            try:
                statement = PeriodFinancialData.from_series(
                    df[col], config_manager=config_manager
                )
            except CouldNotParseException:
                raise CouldNotParseException(
                    "Passed DataFrame did not have any statement items in the index. "
                    "Did you set the column with statement items to the index? Got index:",
                    df.index,
                )
            statement_date = pd.to_datetime(col)
            statements_dict[statement_date] = statement

        if disp_unextracted:
            # Warn about unextracted names
            all_unextracted_names = set()
            for stmt_data in statements_dict.values():
                all_unextracted_names.update(stmt_data.unextracted_names)
            if all_unextracted_names:
                logger.info(
                    f"Was not able to extract data from the following names: {all_unextracted_names}"
                )

        return cls(statements_dict)

    def to_df(self) -> pd.DataFrame:
        all_series = []
        for date, statement in self.statements.items():
            series = statement.to_series()
            series.name = date
            all_series.append(series)
        return pd.concat(all_series, axis=1)

    @property
    def _formatted_df(self) -> pd.DataFrame:
        out_df = self.df.copy()
        out_df.columns = [col.strftime("%m/%d/%Y") for col in out_df.columns]
        return out_df.applymap(lambda x: f"${x:,.0f}" if not x == 0 else " - ")

    def _forecast(
        self, statements, **kwargs
    ) -> Tuple[Dict[str, Forecast], Dict[str, pd.Series]]:
        if "freq" not in kwargs:
            freq = self.freq
            if freq is None:
                raise MixedFrequencyException(
                    "Could not automatically determine frequency of history. Likely there are mixed "
                    "frequencies in the data. Either pass an explicit freq to forecast or remove the "
                    "periods which do not match the frequency before running the forecast."
                )
            kwargs[
                "freq"
            ] = freq  # use historical frequency if desired frequency not passed

        forecast_config = ForecastConfig(**kwargs)
        forecast_dict: Dict[str, Forecast] = {}
        results: Dict[str, pd.Series] = {}
        logger.info(f"Forecasting {self.statement_name}")
        item: ItemConfig
        for item in tqdm(self.config.items):
            if not item.forecast_config.make_forecast:
                # If user set to skip the forecast, skip it as well
                # By default, all calculated items will be skipped
                continue
            data = getattr(statements, item.key)
            pct_of_series = None
            pct_of_config = None
            if item.forecast_config.pct_of is not None:
                pct_of_series = getattr(statements, item.forecast_config.pct_of)
                pct_of_config = statements.config.get(item.forecast_config.pct_of)
            forecast = Forecast(
                data,
                forecast_config,
                item.forecast_config,
                item,
                pct_of_series=pct_of_series,
                pct_of_config=pct_of_config,
            )
            forecast.fit()
            forecast.predict()
            forecast_dict[item.key] = forecast
            if forecast.result is not None:
                forecast.result.name = item.primary_name
            if item.forecast_config.pct_of is not None:
                key_pct_of_key = _key_pct_of_key(item.key, item.forecast_config.pct_of)
                results[key_pct_of_key] = forecast.result
            else:
                results[item.key] = forecast.result

        return forecast_dict, results

    @property
    def freq(self) -> str:
        return pd.infer_freq(self.dates)

    @property
    def dates(self) -> List[pd.Timestamp]:
        return list(self.statements.keys())

    def item_is_empty(self, key: str) -> bool:
        item: pd.Series = getattr(self, key)
        return item_series_is_empty(item)

    def __add__(self, other):
        if isinstance(other, (float, int)):
            new_df = self.df + other
        elif isinstance(other, FinStatementsBase):
            new_df = combine_statement_dfs(self.df, other.df, operation=operator.add)
        else:
            raise NotImplementedError(
                f"cannot add type {type(other)} to type {type(self)}"
            )

        # TODO [#42]: combined statements retain only item config of first statements
        #
        # Think about the best way to handle this. This applies to all math dunder methods.
        new_statements = type(self).from_df(
            new_df, self.config.items, disp_unextracted=False
        )
        return new_statements

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            new_df = self.df * other
        elif isinstance(other, FinStatementsBase):
            new_df = combine_statement_dfs(self.df, other.df, operation=operator.mul)
        else:
            raise NotImplementedError(
                f"cannot multiply type {type(other)} to type {type(self)}"
            )

        new_statements = type(self).from_df(
            new_df, self.config.items, disp_unextracted=False
        )
        return new_statements

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            new_df = self.df - other
        elif isinstance(other, FinStatementsBase):
            new_df = combine_statement_dfs(self.df, other.df, operation=operator.sub)
        else:
            raise NotImplementedError(
                f"cannot subtract type {type(other)} to type {type(self)}"
            )

        new_statements = type(self).from_df(
            new_df, self.config.items, disp_unextracted=False
        )
        return new_statements

    def __rsub__(self, other):
        return (-1 * self) + other

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            new_df = self.df / other
        elif isinstance(other, FinStatementsBase):
            new_df = combine_statement_dfs(
                self.df, other.df, operation=operator.truediv
            )
        else:
            raise NotImplementedError(
                f"cannot divide type {type(other)} to type {type(self)}"
            )

        new_statements = type(self).from_df(
            new_df, self.config.items, disp_unextracted=False
        )
        return new_statements

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            new_df = other / self.df
        else:
            raise NotImplementedError(
                f"cannot divide type {type(other)} to type {type(self)}"
            )

        new_statements = type(self).from_df(
            new_df, self.config.items, disp_unextracted=False
        )
        return new_statements

    def __round__(self, n=None) -> "FinStatementsBase":
        new_df = round(self.df, n)
        new_statements = type(self).from_df(
            new_df, self.config.items, disp_unextracted=False
        )
        return new_statements


def combine_statement_dfs(
    df: pd.DataFrame,
    df2: pd.DataFrame,
    operation: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] = operator.add,
) -> pd.DataFrame:
    common_cols = [col for col in df.columns if col in df2.columns]
    df_unique_cols = [col for col in df.columns if col not in df2.columns]
    df2_unique_cols = [col for col in df2.columns if col not in df.columns]
    common_df = operation(df[common_cols], df2[common_cols])
    result = pd.concat([common_df, df[df_unique_cols], df2[df2_unique_cols]], axis=1)
    cols = sorted(list(result.columns))
    return result[cols]
