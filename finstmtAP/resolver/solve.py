from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sympy import Eq, Expr, Indexed, IndexedBase, Symbol, expand, sympify
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

from finstmt.items.config import ItemConfig

PLUG_SCALE = 1e11


def sympy_dict_to_results_dict(
    s_dict: Dict[IndexedBase, float],
    forecast_dates: pd.DatetimeIndex,
    item_configs: List[ItemConfig],
    t_offset: int = 0,
) -> Dict[str, pd.Series]:
    item_config_dict: Dict[str, ItemConfig] = {
        config.key: config for config in item_configs
    }
    new_results = {}
    for expr in s_dict:
        key = str(expr.base)  # type: ignore[attr-defined]
        try:
            config = item_config_dict[key]
        except KeyError:
            # Must be pct of item, don't need in final results
            continue
        new_results[key] = pd.Series(
            index=forecast_dates, dtype="float", name=config.primary_name
        )
    for expr, val in s_dict.items():
        key = str(expr.base)  # type: ignore[attr-defined]
        t = int(expr.indices[0]) - t_offset  # type: ignore[attr-defined]
        if t < 0:
            # Don't need to store historical results
            continue
        if key not in new_results:
            # Pct of item, skip it, don't need in final results
            continue
        new_results[key].iloc[t] = float(val)
    return new_results


def results_dict_to_sympy_dict(
    results_dict: Dict[str, pd.Series], sympy_namespace: Dict[str, Expr]
) -> Dict[IndexedBase, float]:
    out_dict = {}
    for key, series in results_dict.items():
        arr = series.values
        for i, val in enumerate(arr):
            t_str = f"{key}[{i + 1}]"
            lhs = sympify(t_str, locals=sympy_namespace)
            out_dict[lhs] = val
    return out_dict


def get_solve_eqs_and_full_subs_dict(
    eqs_for_sub: List[Eq], subs_dict: Dict[IndexedBase, float]
) -> Tuple[List[Eq], Dict[IndexedBase, float]]:
    subbed_eqs = []
    subs_dict = subs_dict.copy()
    finished = False
    while not finished:
        next_eqs_to_sub = []
        for eq in eqs_for_sub:
            subbed = Eq(eq.lhs, eq.rhs.xreplace(subs_dict))
            if isinstance(subbed, (BooleanFalse, BooleanTrue)):
                # Both lhs and rhs was subbed. This means it is a calculated item which was
                # added to the forecast because it is being used as pct_of in other items
                # Simply don't solve the equation again as the result is already in the subs dict
                continue
            elif subbed.rhs.free_symbols == set():
                # Equation is completely solved
                subbed_eqs.append(subbed)
                subs_dict[subbed.lhs] = subbed.rhs  # use it in other solutions
            else:
                # Equation still has remaining symbols, handle on next loop
                next_eqs_to_sub.append(subbed)
        if not next_eqs_to_sub:
            finished = True
        if eqs_for_sub == next_eqs_to_sub:
            # Went through a loop without any equations changed.
            # Not going to make any more progress, so exit without complete substitution
            finished = True
        eqs_for_sub = next_eqs_to_sub.copy()
    return eqs_for_sub, subs_dict


def solve_equations(
    solve_eqs: List[Eq],
    subs_dict: Dict[IndexedBase, float],
    substitute: bool = True,
    round_results: bool = True,
):
    solutions_dict = subs_dict.copy()

    if substitute:
        solve_eqs, solutions_dict = get_solve_eqs_and_full_subs_dict(
            solve_eqs, solutions_dict
        )
    solve_exprs = []
    to_solve_for = []
    for eq in solve_eqs:
        solve_exprs.append(eq.rhs - eq.lhs)
        to_solve_for.append(eq.lhs)
    to_solve_for = list(set(to_solve_for))

    res_set = numpy_solve(solve_exprs, to_solve_for)
    if not res_set:
        raise ValueError("could not solve equations")
    solutions_dict.update(res_set)

    return solutions_dict


def _solve_eqs_with_plug_solutions(
    eqs: List[Eq],
    plug_solutions: Dict[IndexedBase, float],
    subs_dict: Dict[IndexedBase, float],
    forecast_dates: pd.DatetimeIndex,
    item_configs: List[ItemConfig],
) -> Dict[IndexedBase, float]:
    subs_dict = subs_dict.copy()
    subs_dict.update(plug_solutions)
    sub_eqs = [Eq(lhs, rhs) for lhs, rhs in subs_dict.items()]
    solutions_dict = solve_equations(eqs + sub_eqs, {}, substitute=False)

    return solutions_dict


def _x_arr_to_plug_solutions(
    x: np.ndarray, plug_keys: Sequence[str], sympy_namespace: Dict[str, IndexedBase]
) -> Dict[IndexedBase, float]:
    x_arrs = np.split(x * PLUG_SCALE, len(plug_keys))
    plug_dict = {key: pd.Series(x_arrs[i]) for i, key in enumerate(plug_keys)}
    # TODO: Is Expr or IndexedBase the correct type?
    plug_solutions = results_dict_to_sympy_dict(plug_dict, sympy_namespace)  # type: ignore[arg-type]
    return plug_solutions


def _symbolic_to_matrix(exprs: Sequence[Expr], variables: Sequence[Symbol]):
    """
    Expr should be in the format of eq.lhs - eq.rhs

    Assuming that there exists numeric matrix A such that equation F = 0
    is equivalent to linear equation Ax = b, this function returns
    tuple (A, b)
    """
    A = []
    b = []
    for expr in exprs:
        coeffs = expand(expr).as_coefficients_dict()
        A.append([float(coeffs[x]) for x in variables])
        b.append(-float(coeffs[1]))
    return np.array(A), np.array(b)


def numpy_solve(exprs: Sequence[Expr], variables: Sequence[Symbol]):
    a_arr, b_arr = _symbolic_to_matrix(exprs, variables)
    x = np.linalg.solve(a_arr, b_arr)
    solution_dict = {var: x[i] for i, var in enumerate(variables)}
    return solution_dict


def _get_indexed_symbols(expr: Union[Eq, Expr]) -> List[Indexed]:
    return [sym for sym in expr.free_symbols if isinstance(sym, Indexed)]
