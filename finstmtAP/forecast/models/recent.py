from typing import Optional

import numpy as np
import pandas as pd

from finstmt.exc import ForecastNotFitException
from finstmt.forecast.models.base import ForecastModel


class RecentValueModel(ForecastModel):
    recent: Optional[float] = None

    def fit(self, series: pd.Series):
        self.recent = series.iloc[-1]
        super().fit(series)

    def predict(self) -> pd.Series:
        if self.orig_series is None:
            raise ForecastNotFitException("call .fit before .predict")
        all_dates = np.concatenate((self.orig_series.index, self._future_date_range))
        df = pd.DataFrame(index=all_dates)
        df["mean"] = self.recent
        self.result_df = df
        self.result = df["mean"].loc[self._future_date_range]
        super().predict()
        return self.result
