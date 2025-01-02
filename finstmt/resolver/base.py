from typing import Dict, List

from sympy import Eq, Idx, IndexedBase

from finstmt.findata.statements import FinancialStatements
from finstmt.resolver.solve import get_solve_eqs_and_full_subs_dict


class ResolverBase:
    solve_eqs: List[Eq]
    subs_dict: Dict[IndexedBase, float]

    def __init__(
        self,
        stmts: FinancialStatements,
    ):
        self.stmts = stmts
        self.global_sympy_namespace = stmts.global_sympy_namespace

        self.set_solve_eqs_and_full_subs_dict()

    # TODO: I need to better understand what's happening here.
    # I think I might have relplicates similar functionality in
    # the FinancialStatements class
    def set_solve_eqs_and_full_subs_dict(self):
        """
        Initialize solving equations and substitution dictionary for financial calculations.
        
        Gets all equations from all_eqs and substitutes known values from sympy_subs_dict.
        Attempts to solve as many equations as possible through substitution and stores 
        both the remaining unsolved equations and the dictionary of solved values.

        Sets:
            solve_eqs: List[Eq] - Equations that still need to be solved
            subs_dict: Dict[IndexedBase, float] - Dictionary of known/solved values

        Example:
            >>> # Given:
            >>> # all_eqs = [
            >>> #    Eq(net_income[1], revenue[1] - costs[1]),
            >>> #    Eq(costs[1], revenue[1] * 0.6)
            >>> # ]
            >>> # sympy_subs_dict = {revenue[1]: 1000}
            >>> resolver.set_solve_eqs_and_full_subs_dict()
            >>> resolver.subs_dict
            {
                revenue[1]: 1000,    # From sympy_subs_dict
                costs[1]: 600        # Solved: 1000 * 0.6
            }
            >>> resolver.solve_eqs
            [
                Eq(net_income[1], 400)  # Remaining equation to solve
            ]
        """
        self.solve_eqs, self.subs_dict = get_solve_eqs_and_full_subs_dict(
            self.all_eqs, self.sympy_subs_dict
        )

    @property
    def t(self) -> Idx:
        return self.stmts.config.sympy_namespace["t"]

    def to_statements(self) -> FinancialStatements:
        raise NotImplementedError

    @property
    def t_indexed_eqs(self) -> List[Eq]:
        raise NotImplementedError

    @property
    def all_eqs(self) -> List[Eq]:
        raise NotImplementedError

    @property
    def num_periods(self) -> int:
        raise NotImplementedError

    @property
    def sympy_subs_dict(self) -> Dict[IndexedBase, float]:
        raise NotImplementedError
