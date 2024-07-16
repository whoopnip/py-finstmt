import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from finstmt.forecast.config import ForecastConfig, ForecastItemConfig
from finstmt.forecast.dataframe import add_cap_and_floor_to_df
from finstmt.forecast.models.base import ForecastModel
from finstmt.items.config import ItemConfig


class FBProphetModel(ForecastModel):
    def __init__(
        self,
        config: ForecastConfig,
        item_config: ForecastItemConfig,
        base_config: ItemConfig,
    ):
        super().__init__(config, item_config, base_config)
        Prophet = _try_import_prophet()

        all_kwargs = {}
        if config.freq.lower() == "y":
            all_kwargs["yearly_seasonality"] = False
        all_kwargs.update(item_config.prophet_kwargs)
        all_kwargs.update(config.prophet_kwargs)
        model = Prophet(**all_kwargs)
        self.model = model

    def fit(self, series: pd.Series):
        self.model.fit(self._df_for_fit(series))
        super().fit(series)

    def predict(self) -> pd.Series:
        future = self.model.make_future_dataframe(**self.config.make_future_df_kwargs)
        add_cap_and_floor_to_df(future, self.item_config.cap, self.item_config.floor)
        forecast = self.model.predict(future)
        self.result_df = forecast
        result = forecast[["ds", "yhat"]].set_index("ds")["yhat"]
        result = result[result.index > self.last_historical_period]
        self.result = result
        super().predict()
        return result

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

        fig = self.model.plot(
            self.result_df, ax=ax, figsize=figsize, xlabel=xlabel, ylabel=ylabel
        )
        if ax is not None and title:
            ax.set_title(title)
        elif title:
            _set_title_on_axes(fig, title)

        plt.close()
        return fig

    def _df_for_fit(self, series: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(series).reset_index()
        df.columns = ["ds", "y"]
        add_cap_and_floor_to_df(df, self.item_config.cap, self.item_config.floor)

        return df


def _try_import_prophet():
    try:
        from prophet import Prophet

    except ImportError:
        raise ImportError(
            "need to install prophet to use forecasting functionality with method auto. "
            "see https://facebook.github.io/prophet/docs/installation.html"
        )

    # Suppress excessive logging from prophet
    prophet_logger = logging.getLogger("prophet")
    prophet_logger.setLevel(logging.WARN)

    return Prophet


def _set_title_on_axes(fig: plt.Figure, title: str):
    for ax in fig.axes:
        ax.set_title(title)
