from typing import Dict, List

import pandas as pd
from sympy import Eq, IndexedBase, sympify

from finstmt.combined.statements import FinancialStatements
from finstmt.resolver.base import ResolverBase
from finstmt.resolver.solve import solve_equations, sympy_dict_to_results_dict


class StatementsResolver(ResolverBase):
    def to_statements(self, **kwargs) -> FinancialStatements:
        if self.solve_eqs:
            solutions_dict = solve_equations(self.solve_eqs, self.subs_dict)
        else:
            solutions_dict = self.subs_dict

        new_results = sympy_dict_to_results_dict(
            solutions_dict,
            pd.DatetimeIndex(self.stmts.dates),
            self.stmts.all_config_items,
        )

        all_results = pd.concat(list(new_results.values()), axis=1).T
        inc_df = self.stmts.income_statements.__class__.from_df(
            all_results,
            self.stmts.income_statements.config.items,
            disp_unextracted=False,
        )
        bs_df = self.stmts.balance_sheets.__class__.from_df(
            all_results, self.stmts.balance_sheets.config.items, disp_unextracted=False
        )

        obj = FinancialStatements(inc_df, bs_df, calculate=False, **kwargs)
        return obj

    @property
    def t_indexed_eqs(self) -> List[Eq]:
        config_managers = [
            self.stmts.income_statements.config.items,
            self.stmts.balance_sheets.config.items,
        ]
        all_eqs = []
        for config_manage in config_managers:
            for config in config_manage:
                lhs = sympify(
                    config.key + "[t]", locals=self.stmts.config.sympy_namespace
                )
                if config.expr_str is not None:
                    rhs = self.stmts.config.expr_for(config.key)
                    eq = Eq(lhs, rhs)
                    all_eqs.append(eq)
        return all_eqs

    @property
    def all_eqs(self) -> List[Eq]:
        t_eqs = self.t_indexed_eqs
        out_eqs = []
        subs_dict = self.sympy_subs_dict
        for period in range(self.num_periods):
            this_t_eqs = []
            for eq in t_eqs:
                period_eq = eq.subs({self.t: period})
                if period_eq.lhs in subs_dict:
                    # Already have data for this, no need to calculate
                    continue
                this_t_eqs.append(period_eq)
            out_eqs.extend(this_t_eqs)
        return out_eqs

    @property
    def num_periods(self) -> int:
        return len(self.stmts.dates)

    @property
    def sympy_subs_dict(self) -> Dict[IndexedBase, float]:
        nper = self.num_periods
        subs_dict = {}
        for config in self.stmts.all_config_items:
            key = config.key
            for period in range(nper):
                t_key = f"{key}[{period}]"
                lhs = sympify(t_key, locals=self.stmts.config.sympy_namespace)
                value = getattr(self.stmts, key).iloc[period]
                if config.expr_str is not None and value == 0:
                    # Don't have a value but it can be calculated, calculate it by not adding to substitutions
                    continue
                subs_dict[lhs] = value
        return subs_dict
