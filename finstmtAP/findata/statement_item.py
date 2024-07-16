from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from sympy import sympify

from finstmt.items.config import ItemConfig

if TYPE_CHECKING:
    from finstmt.findata.period_data import PeriodFinancialData


@dataclass
class StatementItem:
    item_config: ItemConfig
    value: Optional[float]

    def __post_init__(self) -> None:
        if (
            self.item_config.force_positive
            and self.item_config.extract_names is not None
        ):
            # If extracted and need to force positive, take absolute value
            if self.value is None:
                return
            positive_value = abs(self.value)
            self.value = positive_value

    def get_value(self, fin_data: "PeriodFinancialData") -> np.float64:
        # if specific value was provided, then return that even if it's a calculated field
        if self.value is not None:
            return np.float64(self.value)

        expr_str = self.item_config.expr_str

        if expr_str is None:
            return np.float64(0)

        ns_syms = fin_data.config_manager.sympy_namespace
        sym_expr = sympify(expr_str, locals=ns_syms)
        sub_list = []
        t = ns_syms["t"]
        for ns_sym in ns_syms.values():
            if ns_sym == t:
                continue
            if ns_sym[t] in sym_expr.free_symbols:
                sub_list.append((ns_sym[t], getattr(fin_data, str(ns_sym))))

        return np.float64(sym_expr.subs(sub_list))
