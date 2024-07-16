from typing import Type

from finstmt.forecast.config import ForecastConfig, ForecastItemConfig
from finstmt.forecast.models.average import AverageModel
from finstmt.forecast.models.base import ForecastModel
from finstmt.forecast.models.cagr import CAGRModel
from finstmt.forecast.models.manual import ManualForecastModel
from finstmt.forecast.models.prophet import FBProphetModel
from finstmt.forecast.models.recent import RecentValueModel
from finstmt.forecast.models.trend import LinearTrendModel
from finstmt.items.config import ItemConfig


def get_model(
    config: ForecastConfig, item_config: ForecastItemConfig, base_config: ItemConfig
) -> ForecastModel:
    model_class: Type[ForecastModel]
    if item_config.method == "auto":
        model_class = FBProphetModel
    elif item_config.method == "trend":
        model_class = LinearTrendModel
    elif item_config.method == "cagr":
        model_class = CAGRModel
    elif item_config.method == "mean":
        model_class = AverageModel
    elif item_config.method == "recent":
        model_class = RecentValueModel
    elif item_config.method == "manual":
        model_class = ManualForecastModel
    else:
        raise NotImplementedError(f"need to implement method {item_config.method}")

    return model_class(config, item_config, base_config)
