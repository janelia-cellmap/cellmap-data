"""Error message templates and utilities for consistent error handling across CellMap-Data."""


# Standard error message templates for parameter validation
class ErrorMessages:
    """Centralized error message templates for consistent error handling."""

    # Parameter validation templates
    REQUIRED_PARAMETER = "Required parameter '{parameter}' was not provided"
    INVALID_PARAMETER_TYPE = (
        "Parameter '{parameter}' must be of type {expected_type}, got {actual_type}"
    )
    INVALID_PARAMETER_VALUE = "Parameter '{parameter}' has invalid value: {value}"
    CONFLICTING_PARAMETERS = "Cannot specify both '{param1}' and '{param2}' parameters"

    # File and path error templates
    FILE_NOT_FOUND = "File not found: {path}"
    INVALID_PATH = "Invalid path: {path}"

    # Data validation templates
    INVALID_ARRAY_SHAPE = (
        "Array '{name}' has invalid shape {shape}, expected {expected_shape}"
    )
    EMPTY_DATA = "No valid data found in {source}"

    # Transform and processing templates
    UNKNOWN_TRANSFORM = "Unknown transform type: {transform}"
    INVALID_TENSOR_STATE = "Tensor contains {issue}: {details}"

    @staticmethod
    def format_required_parameter(parameter: str) -> str:
        """Format a required parameter error message."""
        return ErrorMessages.REQUIRED_PARAMETER.format(parameter=parameter)

    @staticmethod
    def format_conflicting_parameters(param1: str, param2: str) -> str:
        """Format a conflicting parameters error message."""
        return ErrorMessages.CONFLICTING_PARAMETERS.format(param1=param1, param2=param2)

    @staticmethod
    def format_invalid_value(parameter: str, value) -> str:
        """Format an invalid parameter value error message."""
        return ErrorMessages.INVALID_PARAMETER_VALUE.format(
            parameter=parameter, value=value
        )
