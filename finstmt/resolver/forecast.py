import itertools
import timeit
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from sympy import Eq, Expr, IndexedBase, solve, sympify
from sympy.core.numbers import NaN

from finstmt.findata.statements import FinancialStatements
from finstmt.config_manage.data import _key_pct_of_key
from finstmt.config_manage.statements import StatementsConfigManager
from finstmt.exc import (
    BalanceSheetNotBalancedException,
    InvalidBalancePlugsException,
    InvalidForecastEquationException,
    MissingDataException,
)
from finstmt.findata.statement_series import StatementSeries
from finstmt.forecast.forecast_item_series import ForecastItemSeries
from finstmt.forecast.statements import ForecastedFinancialStatements
from finstmt.findata.item_config import ItemConfig
from finstmt.logger import logger
from finstmt.resolver.base import ResolverBase

# TODO [#46]: clean up ForecastResolver
#
# `ForecastResolver` and associated logic is messy after reworking it multiple times.
# Need to remove unneeded code and restructure more logic into classes. `PlugResult`
# could handle more operations with the plugs, and the math could be more separated
# from the finance logic.
from finstmt.resolver.solve import (
    PLUG_SCALE,
    _get_indexed_symbols,
    _solve_eqs_with_plug_solutions,
    _symbolic_to_matrix,
    _x_arr_to_plug_solutions,
    solve_equations,
    sympy_dict_to_results_dict,
)


class ForecastResolver(ResolverBase):
    def __init__(
        self,
        stmts: "FinancialStatements",
        forecast_dict: Dict[str, ForecastItemSeries],
        bs_diff_max: float,
        timeout: float,
        balance: bool = True,
    ):
        self.forecast_dict = forecast_dict
        self.bs_diff_max = bs_diff_max
        self.timeout = timeout
        self.balance = balance

        if balance:
            self.exclude_plugs = True
        else:
            self.exclude_plugs = False

        super().__init__(stmts)

    def resolve_balance_sheet(self):
        logger.info("Balancing balance sheet")

        solutions_dict = self.subs_dict.copy()
        new_solutions = resolve_balance_sheet(
            self.plug_x0,
            self.solve_eqs,
            self.plug_keys,
            self.subs_dict,
            self.forecast_dates,
            self.stmts.config,
            self.stmts.config.sympy_namespace,
            self.bs_diff_max,
            self.stmts.config.balance_groups,
            self.timeout,
        )
        solutions_dict.update(new_solutions)

        return solutions_dict

    def to_statements(self) -> ForecastedFinancialStatements:
        if self.balance:
            solutions_dict = self.resolve_balance_sheet()
        else:
            if self.solve_eqs:
                solutions_dict = solve_equations(self.solve_eqs, self.subs_dict)
            else:
                solutions_dict = self.subs_dict

        # print(solutions_dict)
        new_results = sympy_dict_to_results_dict(
            solutions_dict, self.forecast_dates, self.stmts.all_config_items, t_offset=1
        )

        if self.balance:
            # Update forecast dict for plug values
            for config in self.plug_configs:
                self.forecast_dict[config.key].to_manual(
                    use_levels=True, replacements=new_results[config.key].values
                )

        all_results = pd.concat(list(new_results.values()), axis=1).T

        stmt_dfs = {}
        for stmt in self.stmts.statements.values():
            stmt_df = StatementSeries.from_df(
                all_results,
                stmt.statement_name,
                stmt.config.items,
                disp_unextracted=False,
            )
            stmt_dfs[stmt.statement_name] = stmt_df

        # type ignore added because for some reason mypy is not picking up structure
        # correctly since it is a dataclass
        # the forecasts passed are just used for plotting
        obj = ForecastedFinancialStatements(stmt_dfs, forecasts=self.forecast_dict, calculate=False)  # type: ignore
        return obj

    @property
    def t_indexed_eqs(self) -> List[Eq]:
        """
        Generate time-indexed equations for each calculated financial statement item.
        
        Returns a list of SymPy equations where variables are indexed with 't' to 
        represent time periods. For each calculated item, generates either:
        1. The equation defined in expr_str if it exists
        2. A percentage equation if pct_of is defined and make_forecast is True
        3. The identity equation (lhs = lhs) otherwise

        Returns:
            List[Eq]: List of SymPy equations with time index 't'

        Examples:
            >>> # Given configuration:
            >>> stmts.config.items = [
            ...     ItemConfig(
            ...         key='net_income',
            ...         expr_str='revenue[t] - expenses[t]'
            ...     ),
            ...     ItemConfig(
            ...         key='cash',
            ...         forecast_config=ForecastItemConfig(
            ...             pct_of='revenue',
            ...             make_forecast=True
            ...         )
            ...     )
            ... ]
            >>> resolver = ForecastResolver(stmts, forecast_dict, bs_diff_max=1000, timeout=180)
            >>> resolver.t_indexed_eqs
            [
                Eq(net_income[t], revenue[t] - expenses[t]),
                Eq(cash[t], revenue[t] * cash_pct_revenue[t])
            ]
            >>> # With use_average=True:
            >>> stmts.config.items = [
            ...     ItemConfig(
            ...         key='interest',
            ...         forecast_config=ForecastItemConfig(
            ...             pct_of='debt',
            ...             make_forecast=True,
            ...             use_average=True
            ...         )
            ...     )
            ... ]
            >>> resolver.t_indexed_eqs
            [
                Eq(interest[t], interest_pct_debt[t] * (debt[t] + debt[t-1])/2)
            ]
        """
        config_managers = []
        for stmt in self.stmts.statements.values():
            config_managers.append(stmt.config.items)
        all_eqs = []
        for config_manage in config_managers:
            for config in config_manage:
                lhs = sympify(
                    config.key + "[t]", locals=self.stmts.config.sympy_namespace
                )
                if config.expr_str is not None:
                    rhs = self.stmts.config.expr_for(config.key)
                elif (
                    config.forecast_config.pct_of is not None
                    and config.forecast_config.make_forecast
                ):
                    key_pct_of_key = _key_pct_of_key(
                        config.key, config.forecast_config.pct_of
                    )
                    if config.forecast_config.use_average:
                        # Use average of current and previous period
                        base = f"({config.forecast_config.pct_of}[t] + {config.forecast_config.pct_of}[t-1])/2"
                    else:
                        base = f"{config.forecast_config.pct_of}[t]"
                    rhs = sympify(
                        f"{base} * {key_pct_of_key}[t]",
                        locals=self.stmts.config.sympy_namespace,
                    )
                else:
                    rhs = lhs  # If my understanding is correct, this means this item is not a calculated item and nothing will be done
                if not rhs == lhs:
                    eq = Eq(lhs, rhs)
                    all_eqs.append(eq)
        return all_eqs

    @property
    def all_eqs(self) -> List[Eq]:
        """
        Generates concrete equations for all time periods by substituting actual time values.
        
        Takes the time-indexed equations from t_indexed_eqs and creates specific equations
        for each forecast period by substituting actual period numbers for 't'. Also handles
        plug values and updates equations based on hardcoded/known values.

        Returns:
            List[Eq]: List of SymPy equations with concrete time values

        Examples:
            >>> # Given t_indexed_eqs with one equation:
            >>> # [Eq(net_income[t], revenue[t] - expenses[t])]
            >>> # And 2 forecast periods:
            >>> resolver.all_eqs
            [
                Eq(net_income[1], revenue[1] - expenses[1]),
                Eq(net_income[2], revenue[2] - expenses[2])
            ]
            >>> # Note: Original equation was expanded into 2 equations,
            >>> # one for each forecast period with t=1 and t=2
        """
        t_eqs = self.t_indexed_eqs
        out_eqs = []
        # Starting from 1 as 0 is last historical period, no need to calculate
        for period in range(1, self.num_periods):
            this_t_eqs = [eq.subs({self.t: period}) for eq in t_eqs]
            out_eqs.extend(this_t_eqs)

        all_hardcoded = _x_arr_to_plug_solutions(
            self.plug_x0, self.plug_keys, self.stmts.config.sympy_namespace
        )
        all_hardcoded.update(self.sympy_subs_dict)
        new_eqs = _get_equations_reformed_for_needed_solutions(
            out_eqs, all_hardcoded, self.stmts.config
        )

        return new_eqs

    @property
    def num_periods(self) -> int:
        # adding 1 because final existing period will be included as period 0
        return list(self.forecast_dict.values())[0].config.periods + 1

    @property
    def forecast_dates(self) -> pd.DatetimeIndex:
        # return list(self.results.values())[0].index
        forecast_item_series = list(self.forecast_dict.values())[0]
        if forecast_item_series.result is not None:
            return forecast_item_series.result.index
        else:
            return forecast_item_series.result_pct.index

    @property
    def sympy_subs_dict(self) -> Dict[IndexedBase, float]:
        """
        Creates a dictionary mapping SymPy indexed expressions to their known values.

        For period 0 (last historical period), gets values from historical data.
        For periods 1+ (forecast periods), gets values from forecast results if available.
        Handles both direct values and percentage-based forecasts.

        Returns:
            Dict[IndexedBase, float]: Dictionary mapping expressions like 'revenue[1]' to values

        Examples:
            >>> # Given:
            >>> # - Last historical: revenue=1000, cash=100
            >>> # - Forecasted: revenue=[1100, 1200]
            >>> # - Cash is forecast as % of revenue at 12%
            >>> stmts.config.items = [
            ...     ItemConfig(key='revenue'),
            ...     ItemConfig(
            ...         key='cash',
            ...         forecast_config=ForecastItemConfig(
            ...             pct_of='revenue',
            ...             make_forecast=True
            ...         )
            ...     )
            ... ]
            >>> resolver.sympy_subs_dict
            {
                revenue[0]: 1000.0,     # Historical value
                revenue[1]: 1100.0,     # Forecasted value
                revenue[2]: 1200.0,     # Forecasted value
                cash[0]: 100.0,         # Historical value 
                cash_pct_revenue[1]: 0.12,  # Forecasted percentage
                cash_pct_revenue[2]: 0.12   # Forecasted percentage
            }
            >>> # Note: cash[1] and cash[2] will be calculated as:
            >>> # cash[1] = revenue[1] * cash_pct_revenue[1] = 1100 * 0.12 = 132
            >>> # cash[2] = revenue[2] * cash_pct_revenue[2] = 1200 * 0.12 = 144
        """
        nper = self.num_periods
        subs_dict = {}
        for config in self.stmts.all_config_items:
            if config.forecast_config.pct_of:
                key = _key_pct_of_key(config.key, config.forecast_config.pct_of)
            else:
                key = config.key

            ### THESE CHANGES WERE PROPOSED, BUT I DON'T THINK ARE NEEDED
            # # Need to include t-1 periods for average calculations
            # start_period = -1 if any(c.forecast_config.use_average for c in self.stmts.all_config_items) else 0
            # for period in range(start_period, nper):

            for period in range(nper):
                t_key = f"{key}[{period}]"
                lhs = sympify(t_key, locals=self.stmts.config.sympy_namespace)
                if period == 0:
                    # period 0 is last historical period, not forecasted period
                    try:
                        value = getattr(self.stmts, key).iloc[-1]
                        # sometimes the value (rhs) can be none. for example, if we have only ONE period in the history
                        # and capex needs to periods in it's definition, then capex[0] will be none. 
                        # we will not include it on the list.
                        if value is None: 
                            continue
                    except AttributeError as e:
                        if "_pct_" in str(e):
                            # Got a percentage of item, only in forecasted results, skip
                            continue
                        else:
                            raise e
                else:
                    # period 1 or later, forecasted period, get from forecast results
                    # If it is a plug item, don't get forecasted values
                    if self.exclude_plugs and config.forecast_config.plug:
                        continue
                    try:
                        # series = self.results[key]
                        if config.forecast_config.pct_of:
                            series = self.forecast_dict[config.key].result_pct
                        else:
                            series = self.forecast_dict[config.key].result
                    except KeyError:
                        # Must not be a forecasted item, probably calculated item
                        continue
                    value = series.iloc[period - 1]
                subs_dict[lhs] = value
            # print("def sympy_subs_dict")
            # print(subs_dict)
        return subs_dict

    @property
    def bs_balance_eqs(self) -> List[Eq]:
        eqs = []
        for balance_set in self.stmts.config.balance_groups:
            for period in range(1, self.num_periods):
                for combo in itertools.combinations(balance_set, 2):
                    lhs_key = f"{combo[0]}[{period}]"
                    lhs = sympify(lhs_key, locals=self.stmts.config.sympy_namespace)
                    rhs_key = f"{combo[1]}[{period}]"
                    rhs = sympify(rhs_key, locals=self.stmts.config.sympy_namespace)
                    eqs.append(Eq(lhs, rhs))
        return eqs

    @property
    def plug_configs(self) -> List[ItemConfig]:
        return [
            conf for conf in self.stmts.all_config_items if conf.forecast_config.plug
        ]

    @property
    def plug_keys(self) -> List[str]:
        return [config.key for config in self.plug_configs]

    @property
    def plug_x0(self) -> np.ndarray:
        x_arrs = []
        for config in self.plug_configs:
            x_arrs.append(self.forecast_dict[config.key].result)
        if len(x_arrs) == 0:  # No plugs
            return []
        x0 = np.concatenate(x_arrs) / PLUG_SCALE
        return x0


@dataclass
class PlugResult:
    res: Optional[np.ndarray] = None
    timeout: float = 180
    start_time: Optional[float] = None
    fun: Optional[float] = None
    met_goal: bool = False

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = timeit.default_timer()

    @property
    def time_elapsed(self) -> float:
        if self.start_time is None:
            raise ValueError("Must instantiate PlugResult to get time_elapsed")
        return timeit.default_timer() - self.start_time

    @property
    def is_timed_out(self) -> bool:
        return self.time_elapsed > self.timeout


def resolve_balance_sheet(
    x0: np.ndarray,
    eqs: List[Eq],
    plug_keys: Sequence[str],
    subs_dict: Dict[IndexedBase, float],
    forecast_dates: pd.DatetimeIndex,
    config: StatementsConfigManager,
    sympy_namespace: Dict[str, IndexedBase],
    bs_diff_max: float,
    balance_groups: List[Set[str]],
    timeout: float,
) -> Dict[IndexedBase, float]:
    """
    Balance the financial statements by adjusting plug values until balance conditions are met.

    Uses numerical optimization to find plug values that make all balance groups equal 
    (e.g., assets = liabilities + equity). Tries to minimize the difference between balance
    groups while maintaining all other financial relationships.

    Args:
        x0: Initial guess for plug values (scaled down by PLUG_SCALE)
        eqs: System of equations defining financial relationships
        plug_keys: Names of variables that can be adjusted to achieve balance
        subs_dict: Known values for variables
        forecast_dates: Dates for which forecasting is being done
        config: Configuration manager containing financial statement structure
        sympy_namespace: Dictionary mapping variable names to SymPy objects
        bs_diff_max: Maximum allowed difference between balance groups
        balance_groups: Groups of items that should be equal (e.g. [{'assets', 'liabilities_and_equity'}])
        timeout: Maximum time in seconds to attempt balancing

    Returns:
        Dictionary mapping variables to their solved values

    Raises:
        BalanceSheetNotBalancedException: If balance cannot be achieved within constraints
        InvalidBalancePlugsException: If plug configuration is invalid
        InvalidForecastEquationException: If equations are inconsistent

    Example:
        >>> # Given initial conditions:
        >>> x0 = np.array([1.0, 1.2])  # Initial guess for cash plug values
        >>> eqs = [
        ...     Eq(assets[1], cash[1] + fixed_assets[1]),
        ...     Eq(liab_and_equity[1], debt[1] + equity[1])
        ... ]
        >>> plug_keys = ['cash', 'debt']  # Cash and debt can be adjusted
        >>> subs_dict = {
        ...     fixed_assets[1]: 1000,
        ...     equity[1]: 800
        ... }
        >>> balance_groups = [{'assets', 'liab_and_equity'}]
        >>> 
        >>> # Resolve balance sheet
        >>> solutions = resolve_balance_sheet(
        ...     x0, eqs, plug_keys, subs_dict, dates, config, 
        ...     namespace, bs_diff_max=1.0, balance_groups=balance_groups,
        ...     timeout=30
        ... )
        >>> solutions
        {
            fixed_assets[1]: 1000,  # From subs_dict
            equity[1]: 800,         # From subs_dict
            cash[1]: 200,          # Solved plug value
            debt[1]: 400,          # Solved plug value
            assets[1]: 1200,       # Calculated: 200 + 1000
            liab_and_equity[1]: 1200  # Calculated: 400 + 800
        }
        >>> # Note: assets = liabilities + equity (1200 = 1200)
    """
    plug_solutions = _x_arr_to_plug_solutions(x0, plug_keys, sympy_namespace)
    all_to_solve: Dict[IndexedBase, Expr] = {}
    for eq in eqs:
        expr = eq.rhs - eq.lhs
        if expr == NaN():
            raise InvalidForecastEquationException(
                f"got NaN forecast equation. LHS: {eq.lhs}, RHS: {eq.rhs}"
            )
        if eq.lhs in all_to_solve:
            raise InvalidForecastEquationException(
                f"got multiple equations to solve for {eq.lhs}. Already had {all_to_solve[eq.lhs]}, now got {expr}"
            )
        all_to_solve[eq.lhs] = expr
    for sol_dict in [subs_dict, plug_solutions]:
        # Plug solutions second here so that they are at end of array
        for lhs, rhs in sol_dict.items():
            expr = rhs - lhs
            if expr == NaN():
                raise MissingDataException(
                    f"got NaN for {lhs} but that is needed for resolving the forecast"
                )
            if lhs in all_to_solve:
                existing_value = all_to_solve[lhs]
                if isinstance(existing_value, float):
                    had_message = f"forecast/plug value of {existing_value}"
                else:
                    had_message = f"equation of {existing_value}"
                raise InvalidForecastEquationException(
                    f"got forecast/plug value for {lhs} but already had an existing {had_message}, now got {expr}"
                )
            all_to_solve[lhs] = expr
    to_solve_for = list(all_to_solve.keys())
    solve_exprs = list(all_to_solve.values())
    _check_for_invalid_system_of_equations(
        eqs, subs_dict, plug_solutions, to_solve_for, solve_exprs
    )
    # TODO: Is Symbol or IndexedBase the correct type here?
    eq_arrs = _symbolic_to_matrix(solve_exprs, to_solve_for)  # type: ignore[arg-type]

    # Get better initial x0 by adding to appropriate plug
    _adjust_x0_to_initial_balance_guess(
        x0, plug_keys, eq_arrs, forecast_dates, to_solve_for, config, balance_groups
    )

    result = PlugResult(timeout=timeout)
    res: Optional[OptimizeResult] = None
    try:
        res = minimize(
            _resolve_balance_sheet_check_diff,
            x0,
            args=(
                eq_arrs,
                forecast_dates,
                to_solve_for,
                bs_diff_max,
                balance_groups,
                result,
            ),
            bounds=[(0, None) for _ in range(len(x0))],  # all positive
            method="TNC",
            options=dict(
                maxCGit=0,
                maxfun=1000000000,
            ),
        )
    except (BalanceSheetBalancedException, BalanceSheetNotBalancedException):
        pass
    if not result.met_goal:
        if result.fun is None or result.res is None:
            # Mainly for mypy purposes
            raise BalanceSheetNotBalancedException(
                "Unexpected balancing error. Did not evaluate the balancing function even once"
            )
        plug_solutions = _x_arr_to_plug_solutions(
            result.res, plug_keys, sympy_namespace
        )
        avg_error = (result.fun**2 / len(result.res)) ** 0.5
        message = (
            f"final solution {plug_solutions} still could not meet max difference of "
            f"${bs_diff_max:,.0f} within timeout of {result.timeout}s. "
            f"Average difference was ${avg_error:,.0f}.\nIf the make_forecast or plug "
            f"configuration for any items were changed, ensure that changes in {plug_keys} can flow through "
            f"to Total Assets and Total Liabilities and Equity. For example, if make_forecast=True for Total Debt "
            f"and make_forecast=False for ST Debt, then using LT debt as a plug will not work as ST debt will "
            f"go down when LT debt goes up.\nOtherwise, consider "
            f"passing to .forecast a timeout greater than {result.timeout}, "
            f"a bs_diff_max at a value greater than {avg_error:,.0f}, or pass "
            f"balance=False to skip balancing entirely."
        )
        raise BalanceSheetNotBalancedException(message)
    else:
        logger.info(f"Balanced in {result.time_elapsed:.1f}s")
    if result.res is None:
        raise BalanceSheetNotBalancedException(
            "Unexpected balancing error. No result found even though met_goal was True"
        )
    plug_solutions = _x_arr_to_plug_solutions(result.res, plug_keys, sympy_namespace)
    solutions_dict = _solve_eqs_with_plug_solutions(
        eqs, plug_solutions, subs_dict, forecast_dates, config.items
    )
    return solutions_dict


def _resolve_balance_sheet_check_diff(
    x: np.ndarray,
    eq_arrs: Tuple[np.ndarray, np.ndarray],
    forecast_dates: pd.DatetimeIndex,
    solve_for: Sequence[IndexedBase],
    bs_diff_max: float,
    balance_groups: List[Set[str]],
    res: PlugResult,
):
    if res.is_timed_out:
        raise BalanceSheetNotBalancedException

    sol_arr = _eq_arrs_and_x_to_sol_arr(x, eq_arrs)
    norms: List[float] = []
    for balance_group in balance_groups:
        balance_arrs = _balance_group_to_balance_arrs(
            balance_group, sol_arr, solve_for, len(forecast_dates)
        )

        norm = 0.0
        for arr_pair in itertools.combinations(balance_arrs, 2):
            diff = abs(arr_pair[0] - arr_pair[1]).astype(float)
            pair_norm = np.linalg.norm(diff)
            norm += pair_norm  # type: ignore[assignment]
        norms.append(norm)

    desired_norm = np.linalg.norm([bs_diff_max] * len(forecast_dates))
    full_norm = sum(norms)
    res.res = x
    res.fun = full_norm
    logger.debug(f"{res.time_elapsed:.1f}: x: {x * PLUG_SCALE}, norm: {full_norm}")
    if all([norm <= desired_norm for norm in norms]):
        res.met_goal = True
        raise BalanceSheetBalancedException(x)
    return full_norm


def _balance_group_to_balance_arrs(
    balance_group: Set[str],
    sol_arr: np.ndarray,
    solve_for: Sequence[IndexedBase],
    num_periods: int,
) -> List[np.ndarray]:
    balance_arrs: List[np.ndarray] = [
        np.zeros(num_periods) for _ in range(len(balance_group))
    ]
    balance_list = list(balance_group)
    for value, var in zip(sol_arr, solve_for):
        key = str(var.base)  # type: ignore[attr-defined]
        if key in balance_list:
            arr_idx = balance_list.index(key)  # type: ignore[attr-defined]
            t = int(var.indices[0]) - 1  # type: ignore[attr-defined]
            if t >= 0:
                balance_arrs[arr_idx][t] = value
    return balance_arrs


def _eq_arrs_and_x_to_sol_arr(
    x: np.ndarray, eq_arrs: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    A_arr, b_arr = eq_arrs
    b_arr[-len(x) :] = -x * PLUG_SCALE  # plug solutions with new X values
    sol_arr = np.linalg.solve(A_arr, b_arr)
    return sol_arr


def _adjust_x0_to_initial_balance_guess(
    x0: np.ndarray,
    plug_keys: Sequence[str],
    eq_arrs: Tuple[np.ndarray, np.ndarray],
    forecast_dates: pd.DatetimeIndex,
    solve_for: Sequence[IndexedBase],
    config: StatementsConfigManager,
    balance_groups: List[Set[str]],
):
    sol_arr = _eq_arrs_and_x_to_sol_arr(x0, eq_arrs)
    n_periods = len(forecast_dates)
    for balance_group in balance_groups:
        balance_arrs = _balance_group_to_balance_arrs(
            balance_group, sol_arr, solve_for, n_periods
        )
        # Get plug which corresponds to each balance item e.g. find cash for assets
        balance_group_plug_keys: List[Optional[str]] = []
        for balance_item in balance_group:
            possible_plug_keys = config.item_determinant_keys(balance_item)
            plug_key: Optional[str] = None
            for key in possible_plug_keys:
                if config.get(key).forecast_config.plug:
                    plug_key = key  # e.g. cash
                    break
            balance_group_plug_keys.append(plug_key)
        bg_with_arrs = list(zip(balance_group, balance_arrs, balance_group_plug_keys))
        for bg_arr1, bg_arr2 in itertools.combinations(bg_with_arrs, 2):
            bg1, arr1, plug1 = bg_arr1
            bg2, arr2, plug2 = bg_arr2
            # e.g. assets - liabilities and equity
            diff = (arr1 - arr2).astype(float)
            for i, d in enumerate(diff):
                # Handle periods one by one
                if d > 0:
                    # e.g. first period asset is greater than first period liabilities and equity
                    # Therefore adjust by adding to liabilities and equity
                    adjust_side = bg2
                    plug_key = plug2
                else:
                    # e.g. first period asset is less than first period liabilities and equity
                    # Therefore adjust by adding to assets
                    adjust_side = bg1
                    plug_key = plug1

                if plug_key is None:
                    normally_calculated_but_not_keys: List[str] = []
                    for item in config.items:
                        if (
                            item.expr_str is not None
                            and item.forecast_config.make_forecast == True
                        ):
                            normally_calculated_but_not_keys.append(item.key)
                    message = (
                        f"Trying to balance {adjust_side} but no plug affects it. One of the following "
                        f"items must have forecast_config.plug = True so that it can be balanced: "
                        f"{config.item_determinant_keys(adjust_side)}. Current plugs: {plug_keys}. "
                    )
                    if normally_calculated_but_not_keys:
                        message += (
                            f"If you expected one of the plugs to affect {adjust_side} but it is not listed "
                            f"in the possible items, it may be that make_forecast has been set to True for an "
                            f"item which would normally be calculated from your plug, but as make_forecast is True "
                            f"it forecasts it rather than calculating and it cannot flow through. Possible items "
                            f"which are normally calculated but are instead being forecasted due to the config: "
                            f"{normally_calculated_but_not_keys}. Either change the plug to be that same item "
                            f"which is normally calculated but instead is forecasted, or set make_forecast=False "
                            f"for that item."
                        )
                    raise InvalidBalancePlugsException(message)

                # Determine index of array to increment. Array has structure of num plugs * num periods, with
                # plugs in order of plug_keys and periods in order within the plugs
                plug_idx = plug_keys.index(plug_key)
                begin_plug_arr_idx = plug_idx * n_periods
                arr_idx = begin_plug_arr_idx + i
                x0[arr_idx] += abs(d) / PLUG_SCALE


class BalanceSheetBalancedException(Exception):
    pass


def _check_for_invalid_system_of_equations(
    eqs: List[Eq],
    subs_dict: Dict[IndexedBase, float],
    plug_solutions: Dict[IndexedBase, float],
    to_solve_for: List[IndexedBase],
    solve_exprs: List[Expr],
):
    if len(to_solve_for) == len(solve_exprs):
        # Equations seem valid, just return
        return

    # Invalid equations, figure out why
    eq_lhs = {eq.lhs for eq in eqs}
    subs_lhs = {key for key in subs_dict}
    plugs_lhs = {key for key in plug_solutions}
    message = f"Got {len(to_solve_for)} items to solve for with {len(solve_exprs)} equations. "
    eq_subs_overlap = eq_lhs.intersection(subs_lhs)
    if eq_subs_overlap:
        message += f"Got {eq_subs_overlap} which overlap between the equations and the calculated values. "
    eq_plugs_overlap = eq_lhs.intersection(plugs_lhs)
    if eq_plugs_overlap:
        message += f"Got {eq_plugs_overlap} which overlap between the equations and the plug values. "
    subs_plugs_overlap = subs_lhs.intersection(plugs_lhs)
    if subs_plugs_overlap:
        message += f"Got {subs_plugs_overlap} which overlap between the calculated values and the plug values. "
    raise InvalidForecastEquationException(message)


def _get_equations_reformed_for_needed_solutions(
    eqs: Sequence[Eq],
    all_hardcoded: Dict[IndexedBase, float],
    config: StatementsConfigManager,
) -> List[Eq]:
    new_eqs = []
    for eq in eqs:
        if eq.lhs in all_hardcoded:
            # Got a calculated item which has also been set with make_forecast=True or as plug=True
            # Solve the equation to see if there is another variable we can set as the lhs which
            # has make_forecast=False and plug=False
            selected_lhs: Optional[IndexedBase] = None
            for sym in _get_indexed_symbols(eq.rhs):
                if sym not in all_hardcoded:
                    selected_lhs = sym  # type: ignore[assignment]
            if selected_lhs is None:
                # Invalid forecast, need to display useful message to the user to fix it.
                # Need to get the original unsubbed equation, as possible variables the user could adjust might
                # have been substituted out of the equation
                key = str(eq.lhs.base)
                orig_expr = config.expr_for(key)
                orig_eq = Eq(eq.lhs, orig_expr)

                possible_fix_strs = []
                possible_symbols = _get_indexed_symbols(orig_eq)
                for sym in possible_symbols:
                    sym_key = str(sym.base)
                    fix_str = (
                        f'\tstmts.config.update("{sym_key}", ["forecast_config", "make_forecast"], False)\n\t'
                        f'stmts.config.update("{sym_key}", ["forecast_config", "plug"], False)'
                    )
                    possible_fix_strs.append(fix_str)
                possible_fix_str = "\nor,\n".join(possible_fix_strs)

                raise InvalidForecastEquationException(
                    f"{eq.lhs} has been set with make_forecast=True or plug=True and yet it is a calculated "
                    f"item. Tried to re-express {orig_eq} in terms of another variable which is not forecasted or "
                    f"plugged but they all are. Set one of {_get_indexed_symbols(orig_eq)} "
                    f"with make_forecast=False and plug=False.\n\nPossible fixes:\n{possible_fix_str}"
                )
            # Another variable in the original equation is not forecasted/plugged. Re-express the equation in
            # terms of that variable
            solution = solve(eq, selected_lhs)[0]
            new_eqs.append(Eq(selected_lhs, solution))
        else:
            new_eqs.append(eq)
    return new_eqs
