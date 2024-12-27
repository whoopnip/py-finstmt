from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from finstmt.findata.item_config import ItemConfig

PAD_PCT = 0.2

@dataclass
class StatementItemSeries:
    """
    Series data for a single financial statement item.
    """
    series: pd.Series
    item_config: ItemConfig

    def plot(
        self, ax: Optional[plt.Axes] = None, figsize: Tuple[int, int] = (12, 5)
    ) -> plt.Figure:
        """
        Plot historical data for this item.
        """
        if ax is None:
            fig = plt.figure(facecolor="w", figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        xlabel = "Time"
        ylabel = None
        title = self.item_config.display_name

        ax.plot(self.series.index, self.series.values, "k.")
        ax.plot(self.series.index, self.series.values, ls="-", c="#0072B2")

        max_point = self.series.values.max()
        min_point = self.series.values.min()
        y_lim_upper = max_point * (1 + PAD_PCT)
        y_lim_lower = min_point * (1 - PAD_PCT)

        ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        plt.close()
        
        return fig
