import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Subplot
from typing_extensions import Self

from finstmt.combined.combinator import ForecastedFinancialStatementsCombinator
from finstmt.combined.statements import FinancialStatements
from finstmt.forecast.main import Forecast

NUM_PLOT_COLUMNS = 3
DEFAULT_WIDTH = 15
DEFAULT_HEIGHT_PER_ROW = 3


@dataclass
class ForecastedFinancialStatements(FinancialStatements):
    forecasts: Dict[str, Forecast] = field(default_factory=lambda: {})

    def __post_init__(self):
        self._combinator = ForecastedFinancialStatementsCombinator()
        super().__post_init__()

    def plot(
        self,
        subset: Optional[Sequence[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        num_cols: int = NUM_PLOT_COLUMNS,
        height_per_row: float = DEFAULT_HEIGHT_PER_ROW,
        plot_width: float = DEFAULT_WIDTH,
    ) -> plt.Figure:
        if subset is not None:
            plot_items = {k: v for k, v in self.forecasts.items() if k in subset}
        else:
            plot_items = self.forecasts

        num_plot_rows = math.ceil(len(plot_items) / num_cols)
        num_plot_columns = min(len(plot_items), num_cols)

        if figsize is None:
            figsize = (plot_width, height_per_row * num_plot_rows)

        fig, axes = plt.subplots(
            num_plot_rows, num_plot_columns, sharex=False, sharey=False, figsize=figsize
        )
        row = 0
        col = 0
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="Attempting to set identical bottom == top"
            )
            for i, (item_key, forecast) in enumerate(plot_items.items()):
                selected_ax = _get_selected_ax(
                    axes, row, col, num_plot_rows, num_plot_columns
                )
                forecast.plot(ax=selected_ax)

                # For before final row, don't display x-axis
                if not _is_last_plot_in_col(
                    row, col, num_plot_rows, num_plot_columns, len(plot_items)
                ):
                    selected_ax.get_xaxis().set_visible(False)

                if i == len(plot_items) - 1 or _plot_finished(
                    row, col, num_plot_rows, num_plot_columns
                ):
                    break
                col += 1
                if col == num_plot_columns:
                    row += 1
                    col = 0
        while not _plot_finished(row, col, num_plot_rows, num_plot_columns):
            col += 1
            if col == num_plot_columns:
                row += 1
                col = 0
            fig.delaxes(axes[row][col])
        return fig

    def __round__(self, n=None) -> Self:
        new_fcst = super().__round__(n)
        new_fcst.forecasts = {k: round(v, n) for k, v in self.forecasts.items()}  # type: ignore[call-overload]
        return new_fcst


def _plot_finished(row: int, col: int, max_rows: int, max_cols: int) -> bool:
    return row == max_rows - 1 and col == max_cols - 1


def _get_selected_ax(
    axes: plt.GridSpec, row: int, col: int, num_plot_rows: int, num_plot_columns: int
) -> Subplot:
    if num_plot_rows == num_plot_columns == 1:
        # No array if single row and column
        return axes
    elif num_plot_rows == 1:
        # 1D array if single row
        return axes[col]
    elif num_plot_columns == 1:
        # 1D array if single column
        return axes[row]
    else:
        # 2D array if multiple rows
        return axes[row, col]


def _is_last_plot_in_col(
    row: int, col: int, num_plot_rows: int, num_plot_columns: int, num_plots: int
) -> bool:
    # In last row, automatically last plot in col
    if row == num_plot_rows - 1:
        return True

    # If earlier than next to last row, must not be last plot in rol
    if row != num_plot_rows - 2:
        return False

    # Must be in next to last row. Determine if there is going to be a plot below
    plot_number = row * num_plot_columns + (col + 1)
    if plot_number + num_plot_columns > num_plots:
        # Moving down one row would mean that is more plots than necessary
        return True
    else:
        return False
