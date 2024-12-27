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

