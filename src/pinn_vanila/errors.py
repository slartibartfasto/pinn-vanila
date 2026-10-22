class PINNVanilaError(Exception):
    """Base package error."""


class ConfigurationError(PINNVanilaError, ValueError):
    """A configuration or physical-property file is inconsistent."""


class GeometryError(PINNVanilaError, ValueError):
    """Geometry is invalid or query points are outside the channel."""


class ExtrapolationError(PINNVanilaError, ValueError):
    """A request lies outside the model training domain."""
