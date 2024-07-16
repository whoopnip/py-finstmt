import dataclasses
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, TypeVar

from typing_extensions import Self

from finstmt.forecast.config import ForecastItemConfig

T = TypeVar("T")


@dataclass
class ItemConfig:
    key: str
    display_name: str

    extract_names: Optional[Sequence[str]] = None
    force_positive: bool = True
    forecast_config: ForecastItemConfig = field(
        default_factory=lambda: ForecastItemConfig()
    )
    expr_str: Optional[str] = None

    # The required verbosity level to display this item in the output
    # 0: always display
    # 1: display in the default output
    # 2: display in verbose output
    # 3: display in very verbose output
    # ... and so on
    display_verbosity: int = 1

    # TODO [#19]: add config and logic for whether to take highest priority or add all of matching names
    #
    # When extracting impairment, in Capital IQ data it has Impairment of Goodwill and Asset Writedown,
    # both of which should be included. This is in contrast to most others where only the highest priority
    # key should be selected

    @property
    def primary_name(self) -> str:
        if self.extract_names is None:
            return self.key

        return self.extract_names[0]

    def copy(self, **updates) -> Self:
        return dataclasses.replace(self, **updates)

    def __round__(self, n=None) -> "ItemConfig":
        return _apply_operation_to_item_config(self, n, round)

    def __add__(self, other: T) -> "ItemConfig":
        return _apply_operation_to_item_config(self, other, operator.add)

    def __sub__(self, other: T) -> "ItemConfig":
        return _apply_operation_to_item_config(self, other, operator.sub)

    def __mul__(self, other: T) -> "ItemConfig":
        return _apply_operation_to_item_config(self, other, operator.mul)

    def __truediv__(self, other: T) -> "ItemConfig":
        return _apply_operation_to_item_config(self, other, operator.truediv)


ItemConfigOperationData = ForecastItemConfig


def _apply_operation_to_item_config(
    item_config: ItemConfig,
    other: T,
    func: Callable[[ItemConfigOperationData, T], ItemConfigOperationData],
) -> ItemConfig:
    updates: Dict[str, Any] = {}
    updates["forecast_config"] = func(
        item_config.forecast_config,
        _get_attr_if_needed(other, "forecast_config"),
    )
    return item_config.copy(**updates)


def _get_attr_if_needed(other: Any, attr: str) -> Any:
    if isinstance(other, ItemConfig):
        return getattr(other, attr)
    else:
        return other
