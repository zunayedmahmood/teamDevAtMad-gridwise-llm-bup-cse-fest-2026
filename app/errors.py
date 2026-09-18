class GridWiseError(Exception):
    """Base class for expected application errors."""


class SemanticInputError(GridWiseError):
    pass


class LLMProviderError(GridWiseError):
    pass


class LLMConfigurationError(LLMProviderError):
    pass


class LLMInterpretationError(GridWiseError):
    pass


class DirectiveValidationError(GridWiseError):
    pass


class OptimizationInfeasibleError(GridWiseError):
    pass


class InternalPlanValidationError(GridWiseError):
    pass
