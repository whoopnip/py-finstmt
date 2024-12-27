import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sympy import Indexed, sympify

from finstmt.findata.item_config import ItemConfig


@dataclass
class StatementItem:
    item_config: ItemConfig
    seed_value: Optional[
        float
    ] = None  # explicitly provided value for this statement item
    calculated_value: Optional[np.float64] = field(init=False, default=None)

    def __post_init__(self) -> None:
        # If extracted and need to force positive, take absolute value
        if self.seed_value is None:
            return

        if self.item_config.force_positive:
            positive_value = abs(self.seed_value)
            self.seed_value = positive_value

    @property
    def value(self) -> Optional[np.float64]:
        # if specific value was provided, then return that even if it's a
        # calculated field
        if (self.seed_value is not None) and (not math.isnan(self.seed_value)):
            return np.float64(self.seed_value)

        if self.item_config.expr_str is None:
            return np.float64(0)

        return self.calculated_value

    # Return a tuple for this for this statement item in the form of (lhs, rhs)
    # Where lhs is the t-indexed key and the rhs is the seed value if it exists
    # otherwise the expr_str
    # The result of this will be used to simulatenously solve all expr_strs for all
    # statement items in all statements and periods
    def get_expression_string(self):
        if (self.seed_value is not None) and (not math.isnan(self.seed_value)):
            return (f"{self.item_config.key}[t]", self.seed_value)
        return (f"{self.item_config.key}[t]", self.item_config.expr_str)

    # If this field is a calculated field, then update the calculated statement idem
    # This will be done by solving all calculated fields simultaneously
    def update_statement_item_calculated_value(self, statement_item_value):
        if self.item_config.expr_str is None: 
            return
        self.calculated_value = statement_item_value

