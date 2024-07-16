from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAD_PCT = 0.2


def plot_forecast(
    fcst_df: pd.DataFrame,
    orig_data: np.ndarray,
    orig_dates: pd.DatetimeIndex,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 5),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    if ax is None:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    fcst_t = fcst_df.index.to_pydatetime()
    ax.plot(orig_dates, orig_data, "k.")
    ax.plot(fcst_t, fcst_df["mean"], ls="-", c="#0072B2")
    if "lower" in fcst_df.columns and "upper" in fcst_df.columns:
        ax.fill_between(
            fcst_t, fcst_df["lower"], fcst_df["upper"], color="#0072B2", alpha=0.2
        )

    # Set y-range based only on the mean and orig data and not confidence interval as it can get very large
    max_point = max(fcst_df["mean"].max(), orig_data.max())
    min_point = min(fcst_df["mean"].min(), orig_data.min())
    y_lim_upper = max_point * (1 + PAD_PCT)
    y_lim_lower = min_point * (1 - PAD_PCT)
    ax.set_ylim([y_lim_lower, y_lim_upper])
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    plt.close()

    return fig
