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
    config: ForecastConfig, item_config: ItemConfig
) -> ForecastModel:
    model_class: Type[ForecastModel]
    if item_config.forecast_config.method == "auto":
        model_class = FBProphetModel
    elif item_config.forecast_config.method == "trend":
        model_class = LinearTrendModel
    elif item_config.forecast_config.method == "cagr":
        model_class = CAGRModel
    elif item_config.forecast_config.method == "mean":
        model_class = AverageModel
    elif item_config.forecast_config.method == "recent":
        model_class = RecentValueModel
    elif item_config.forecast_config.method == "manual":
        model_class = ManualForecastModel
    else:
        raise NotImplementedError(f"need to implement method {item_config.forecast_config.method}")

    return model_class(config, item_config)
