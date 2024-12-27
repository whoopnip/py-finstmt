from typing import Type

from finstmt.findata.item_forecast_config import ForecastItemConfig
from finstmt.forecast.config import ForecastConfig
from finstmt.forecast.models.average import AverageModel
from finstmt.forecast.models.base import ForecastModel
from finstmt.forecast.models.cagr import CAGRModel
from finstmt.forecast.models.manual import ManualForecastModel
from finstmt.forecast.models.prophet import FBProphetModel
from finstmt.forecast.models.recent import RecentValueModel
from finstmt.forecast.models.trend import LinearTrendModel
from finstmt.findata.item_config import ItemConfig


def get_model(
    config: ForecastConfig, forecast_item_config: ForecastItemConfig, item_config: ItemConfig
) -> ForecastModel:
    model_class: Type[ForecastModel]
    if forecast_item_config.method == "auto":
        model_class = FBProphetModel
    elif forecast_item_config.method == "trend":
        model_class = LinearTrendModel
    elif forecast_item_config.method == "cagr":
        model_class = CAGRModel
    elif forecast_item_config.method == "mean":
        model_class = AverageModel
    elif forecast_item_config.method == "recent":
        model_class = RecentValueModel
    elif forecast_item_config.method == "manual":
        model_class = ManualForecastModel
    else:
        raise NotImplementedError(f"need to implement method {forecast_item_config.method}")

    return model_class(config, forecast_item_config, item_config)
