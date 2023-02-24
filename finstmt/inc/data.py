from dataclasses import dataclass
from typing import Optional

from finstmt.findata.database import FinDataBase
from finstmt.inc.config import INCOME_STATEMENT_INPUT_ITEMS
from finstmt.exc import NoSuchItemException

@dataclass(unsafe_hash=True)
class IncomeStatementData(FinDataBase):
    data = {}

    items_config = INCOME_STATEMENT_INPUT_ITEMS

    def __getattr__(self, item_key: str):
        """
        Get the Income Statement Value for a given key
        """
        try:
            return self.data.get(item_key)
        except NoSuchItemException:
            raise AttributeError(item_key)

    def __setattr__(self, item_key, value):
        self.data[item_key] = value


    # revenue: Optional[float] = 0
    # cogs: Optional[float] = 0
    # sga: Optional[float] = 0
    # int_exp: Optional[float] = 0
    # tax_exp: Optional[float] = 0
    # rd_exp: Optional[float] = 0
    # dep_exp: Optional[float] = 0
    # other_op_exp: Optional[float] = 0
    # gain_on_sale_invest: Optional[float] = 0
    # gain_on_sale_asset: Optional[float] = 0
    # impairment: Optional[float] = 0

    # op_exp: Optional[float] = None
    # ebit: Optional[float] = None
    # ebt: Optional[float] = None
    # net_income: Optional[float] = None


    # @property
    # def gross_profit(self) -> Optional[float]:
    #     if self.revenue is None or self.cogs is None:
    #         return None
    #     return self.revenue - self.cogs

    # @property
    # def effective_tax_rate(self) -> float:
    #     if self.ebt is None:
    #         raise ValueError("cannot calculate effective tax rate as ebt is None")
    #     elif self.tax_exp is None:
    #         raise ValueError(
    #             "cannot calculate effective tax rate is tax expense is None"
    #         )
    #     return self.tax_exp / self.ebt
