import dataclasses
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import pandas as pd
from typing_extensions import Self

T = TypeVar("T")


@dataclass
class ForecastConfig:
    periods: int = 5
    freq: str = "Y"
    prophet_kwargs: dict = field(default_factory=lambda: {})
    balance: bool = True
    timeout: float = 180

    # TODO [#45]: after handling units, adjust default allowed BS difference for units
    bs_diff_max: float = 10000

    def __post_init__(self):
        if self.freq.casefold() == "y":
            self.freq = "12m"
        elif self.freq.casefold() == "q":
            self.freq = "3m"

    @property
    def make_future_df_kwargs(self) -> Dict[str, Union[int, str]]:
        return dict(periods=self.periods, freq=self.freq)


@dataclass
class ForecastItemConfig:
    """
    The main configuration for a single item forecast

    method: 'auto' currently only supported method, runs forecast with prophet
    pct_of: key of financial statement item to forecast this as a percentage of
    make_forecast: whether to forecast
    prophet_kwargs: kwargs to pass to prophet model
    cap: the maximum that the trend line should reach
    floor: the minimum that the trend line should reach
    manual_forecasts: manually set values to use instead of doing a forecast
    plug: Whether to make this item adjustable to balance the balance sheet
    balance_item: Whether this item is balanced in the balancing process after forecasting
        and which item to balance it with (typically just set to 'total_liab_and_equity' for
        total assets and 'total_assets' for total liabilities and equity)
    """

    method: str = "cagr"
    pct_of: Optional[str] = None
    make_forecast: bool = True
    prophet_kwargs: dict = field(default_factory=lambda: {})
    cap: Optional[Union[float, pd.Series]] = None
    floor: Optional[Union[float, pd.Series]] = None
    manual_forecasts: Dict[str, List[float]] = field(
        default_factory=lambda: {"levels": [], "growth": []}
    )
    plug: bool = False
    balance_with: Optional[str] = None

    def to_series(self) -> pd.Series:
        out_dict = {
            "Method": self.method,
            "% of": self.pct_of,
            "Cap": self.cap,
            "Floor": self.floor,
            "Plug": self.plug,
        }
        out_dict.update(self.prophet_kwargs)
        if self.manual_forecasts["levels"]:
            out_dict.update({"Manual Levels": self.manual_forecasts["levels"]})
        if self.manual_forecasts["growth"]:
            growth_pcts = [
                f"{growth:.2%}" for growth in self.manual_forecasts["growth"]
            ]
            out_dict.update({"Manual Growth": growth_pcts})
        return pd.Series(out_dict)

    def copy(self, **updates) -> Self:
        return dataclasses.replace(self, **updates)

    def __round__(self, n: Optional[int] = None) -> "ForecastItemConfig":
        return _apply_operation_to_item_config(self, n, round)  # type: ignore[misc,arg-type]

    def __add__(self, other: T) -> "ForecastItemConfig":
        return _apply_operation_to_item_config(self, other, operator.add)

    def __sub__(self, other: T) -> "ForecastItemConfig":
        return _apply_operation_to_item_config(self, other, operator.sub)

    def __mul__(self, other: T) -> "ForecastItemConfig":
        return _apply_operation_to_item_config(self, other, operator.mul)

    def __truediv__(self, other: T) -> "ForecastItemConfig":
        return _apply_operation_to_item_config(self, other, operator.truediv)


def _apply_operation_to_item_config(
    item_config: ForecastItemConfig,
    other: T,
    func: Callable[[Any, T], Any],
) -> ForecastItemConfig:
    updates: Dict[str, Any] = {}
    if item_config.cap is not None:
        updates["cap"] = func(item_config.cap, _get_attr_if_needed(other, "cap"))
    if item_config.floor is not None:
        updates["floor"] = func(item_config.floor, _get_attr_if_needed(other, "floor"))
    manual_forecast_keys = ["levels", "growth"]
    updates["manual_forecasts"] = {}
    for key in manual_forecast_keys:
        if item_config.manual_forecasts[key]:
            updates["manual_forecasts"][key] = [
                func(val, _get_manual_forecast_key_if_needed(other, key, i))
                for i, val in enumerate(item_config.manual_forecasts[key])
            ]
        else:
            updates["manual_forecasts"][key] = []
    return item_config.copy(**updates)


def _get_attr_if_needed(other: Any, attr: str) -> Any:
    if isinstance(other, ForecastItemConfig):
        return getattr(other, attr)
    else:
        return other


def _get_manual_forecast_key_if_needed(other: Any, key: str, i: int) -> Any:
    if isinstance(other, ForecastItemConfig):
        return other.manual_forecasts[key][i]
    else:
        return other
