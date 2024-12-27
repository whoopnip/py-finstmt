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
