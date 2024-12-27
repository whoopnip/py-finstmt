import operator
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Protocol, TypeVar

from typing_extensions import TypeGuard

if TYPE_CHECKING:
    from finstmt.findata.statements import FinancialStatements
    from finstmt.findata.statement_series import StatementSeries
    from finstmt.forecast.forecast_item_series import ForecastItemSeries
    from finstmt.forecast.statements import ForecastedFinancialStatements

T = TypeVar("T")
StatementT = TypeVar("StatementT", bound="FinancialStatements")


class StatementsCombinator(Protocol[StatementT]):
    def add(self, statement: StatementT, other: Any) -> StatementT:
        ...

    def subtract(self, statement: StatementT, other: Any) -> StatementT:
        ...

    def multiply(self, statement: StatementT, other: Any) -> StatementT:
        ...

    def divide(self, statement: StatementT, other: Any) -> StatementT:
        ...


class FinancialStatementsCombinator(StatementsCombinator["FinancialStatements"]):
    def add(
        self, statement: "FinancialStatements", other: Any
    ) -> "FinancialStatements":
        statements = _apply_to_child_statements(statement, other, operator.add)
        return statement.copy(statements=statements)

    def subtract(
        self, statement: "FinancialStatements", other: Any
    ) -> "FinancialStatements":
        statements = _apply_to_child_statements(statement, other, operator.sub)
        return statement.copy(statements=statements)

    def multiply(
        self, statement: "FinancialStatements", other: Any
    ) -> "FinancialStatements":
        statements = _apply_to_child_statements(statement, other, operator.mul)
        return statement.copy(statements=statements)

    def divide(
        self, statement: "FinancialStatements", other: Any
    ) -> "FinancialStatements":
        statements = _apply_to_child_statements(statement, other, operator.truediv)
        return statement.copy(statements=statements)


class ForecastedFinancialStatementsCombinator(
    StatementsCombinator["ForecastedFinancialStatements"]
):
    def add(
        self, statement: "ForecastedFinancialStatements", other: Any
    ) -> "ForecastedFinancialStatements":
        statements = _apply_to_child_statements(statement, other, operator.add)
        forecasts = _apply_to_forecasts(statement.forecasts, other, operator.add)
        return statement.copy(
            statements=statements,
            forecasts=forecasts,
        )

    def subtract(
        self, statement: "ForecastedFinancialStatements", other: Any
    ) -> "ForecastedFinancialStatements":
        statements = _apply_to_child_statements(statement, other, operator.sub)
        forecasts = _apply_to_forecasts(statement.forecasts, other, operator.sub)
        return statement.copy(
            statements=statements,
            forecasts=forecasts,
        )

    def multiply(
        self, statement: "ForecastedFinancialStatements", other: Any
    ) -> "ForecastedFinancialStatements":
        statements = _apply_to_child_statements(statement, other, operator.mul)
        forecasts = _apply_to_forecasts(statement.forecasts, other, operator.mul)
        return statement.copy(
            statements=statements,
            forecasts=forecasts,
        )

    def divide(
        self, statement: "ForecastedFinancialStatements", other: Any
    ) -> "ForecastedFinancialStatements":
        statements = _apply_to_child_statements(statement, other, operator.truediv)
        forecasts = _apply_to_forecasts(statement.forecasts, other, operator.truediv)
        return statement.copy(
            statements=statements,
            forecasts=forecasts,
        )


def _apply_to_child_statements(
    statements: "FinancialStatements",
    other: Any,
    func: Callable[[Any, Any], Any],
) -> List["StatementSeries"]:
    from finstmt import FinancialStatements

    if isinstance(other, (float, int)):
        new_stmts = {}
        for statement in statements.statements.values():
            new_stmt = func(statement, other)
            new_stmts[new_stmt.statement_name] = new_stmt
    elif isinstance(other, FinancialStatements):
        new_stmts = {}
        for left, right in zip(statements.statements.values(), other.statements.values()):
            new_stmt = func(left, right)
            new_stmts[new_stmt.statement_name] = new_stmt
    else:
        raise NotImplementedError(
            f"cannot {func.__name__} type {type(statements)} to type {type(other)}"
        )

    return new_stmts


def _apply_to_forecasts(
    forecasts: Dict[str, "ForecastItemSeries"],
    other: Any,
    func: Callable[["ForecastItemSeries", Any], "ForecastItemSeries"],
) -> Dict[str, "ForecastItemSeries"]:
    if isinstance(other, (float, int)):
        return {k: func(v, other) for k, v in forecasts.items()}
    elif _is_forecasted_financial_statements(other):
        return {k: func(v, other.forecasts[k]) for k, v in forecasts.items()}
    else:
        raise NotImplementedError(
            f"cannot {func.__name__} type {type(forecasts)} to type {type(other)}"
        )


def _is_forecasted_financial_statements(
    obj: Any,
) -> TypeGuard["ForecastedFinancialStatements"]:
    return hasattr(obj, "forecasts")
