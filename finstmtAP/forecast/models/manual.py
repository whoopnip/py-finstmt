from typing import Optional

import pandas as pd

from finstmt.exc import ImproperManualForecastException
from finstmt.forecast.config import ForecastConfig, ForecastItemConfig
from finstmt.forecast.models.base import ForecastModel
from finstmt.items.config import ItemConfig


class ManualForecastModel(ForecastModel):
    recent: Optional[float] = None

    def __init__(
        self,
        config: ForecastConfig,
        item_config: ForecastItemConfig,
        base_config: ItemConfig,
    ):
        super().__init__(config, item_config, base_config)
        self._set_growths_levels()
        self._validate()

    def _validate(self):
        if not self.growths and not self.levels:
            raise ImproperManualForecastException(
                "must provide either growth or levels for manual forecast"
            )
        if self.growths and self.levels:
            raise ImproperManualForecastException(
                "must only provide one of growth or levels for manual forecast"
            )

        forecast_length_error_str = (
            f"were provided for {self.config.periods} forecast periods"
        )
        if self.growths:
            if len(self.growths) != self.config.periods:
                raise ImproperManualForecastException(
                    f"{len(self.growths)} growth rates {forecast_length_error_str}"
                )
        else:
            if len(self.levels) != self.config.periods:
                raise ImproperManualForecastException(
                    f"{len(self.levels)} levels {forecast_length_error_str}"
                )

    def _set_growths_levels(self):
        self.growths = self.item_config.manual_forecasts["growth"]
        self.levels = self.item_config.manual_forecasts["levels"]

    def fit(self, series: pd.Series):
        self.recent = series.iloc[-1]
        super().fit(series)

    def predict(self) -> pd.Series:
        if self.growths:
            values = []
            last_value = self.recent
            for growth in self.growths:
                next_value = last_value * (1 + growth)
                values.append(next_value)
                last_value = next_value
        else:
            values = self.levels

        self.result = pd.Series(values, index=self._future_date_range)
        self.result_df = pd.DataFrame(
            pd.concat([self.orig_series, self.result]), columns=["mean"]
        )
        super().predict()
        return self.result
