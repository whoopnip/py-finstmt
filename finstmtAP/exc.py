class FinstmtException(Exception):
    pass


class NotACalculatedItemException(FinstmtException):
    pass


class NoSuchItemException(FinstmtException):
    pass


class CouldNotParseException(FinstmtException):
    pass


class MissingDataException(FinstmtException):
    pass


class MixedFrequencyException(FinstmtException):
    pass


class MismatchingDatesException(FinstmtException):
    pass


class ForecastException(FinstmtException):
    pass


class ForecastNotFitException(ForecastException):
    pass


class ForecastNotPredictedException(ForecastException):
    pass


class ImproperManualForecastException(ForecastException):
    pass


class BalanceSheetNotBalancedException(ForecastException):
    pass


class InvalidForecastEquationException(ForecastException):
    pass


class InvalidBalancePlugsException(ForecastException):
    pass


class InvalidBalanceConfigException(ForecastException):
    pass
