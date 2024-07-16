import datetime
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DatetimeIndex

from finstmt.exc import ForecastNotFitException, ForecastNotPredictedException
from finstmt.forecast.config import ForecastConfig, ForecastItemConfig
from finstmt.forecast.plot import plot_forecast
from finstmt.items.config import ItemConfig


class ForecastModel:
    result: Optional[pd.Series] = None
    result_df: Optional[pd.DataFrame] = None
    last_historical_period: Optional[datetime.datetime] = None
    orig_series: Optional[pd.Series] = None

    def __init__(
        self,
        config: ForecastConfig,
        item_config: ForecastItemConfig,
        base_config: ItemConfig,
    ):
        self.config = config
        self.item_config = item_config
        self.base_config = base_config
        self.has_been_fit = False
        self.has_prediction = False

    def fit(self, series: pd.Series):
        self.last_historical_period = series.index.max()
        self.has_been_fit = True
        self.orig_series = series

    def predict(self) -> pd.Series:
        self.has_prediction = True
        return pd.Series()

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (12, 5),
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        if xlabel is None:
            xlabel = "Time"
        if title is None:
            title = self.base_config.display_name

        if self.orig_series is None:
            raise ForecastNotPredictedException("call .fit then .predict before .plot")

        return plot_forecast(
            self.result_df,
            self.orig_series.values,
            self.orig_series.index,
            ax=ax,
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

    @property
    def _future_date_range(self) -> DatetimeIndex:
        if not self.has_been_fit:
            raise ForecastNotFitException("call .fit before ._future_date_range")
        return pd.date_range(
            start=self.last_historical_period,
            periods=self.config.periods + 1,
            freq=self.config.freq,
            closed="right",
        )

    @property
    def historical_freq(self) -> str:
        if self.orig_series is None:
            raise ForecastNotFitException("call .fit before .historical_freq")
        return pd.infer_freq(self.orig_series.index)

    @property
    def desired_freq_t_multiplier(self) -> float:
        """
        The  multiplier of the forecast frequncy versus the historical frequency. E.g.
        if the forecast is annual and historical is quarterly then the multiplier is 4.

        :return:
        """
        if self.orig_series is None:
            raise ForecastNotFitException("call .fit before .desired_freq_t_multiplier")
        return compare_freq_strs(
            self.config.freq, self.historical_freq, ref_date=self.orig_series.index[-1]
        )


def compare_freq_strs(
    freq1: str,
    freq2: str,
    ref_date: Union[pd.Timestamp, datetime.datetime, str] = "1/1/2000",
) -> float:
    periods = 10
    dates1 = pd.date_range(start=ref_date, freq=freq1, periods=periods)
    td1 = dates1[-1] - dates1[0]
    dates2 = pd.date_range(start=ref_date, freq=freq2, periods=periods)
    td2 = dates2[-1] - dates2[0]
    return td1 / td2
