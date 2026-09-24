class DeciShiftError(Exception):
    """Base exception for DeciShift."""


class ConfigurationError(DeciShiftError):
    """Raised when local configuration is invalid."""


class InsufficientEvidenceError(DeciShiftError):
    """Raised when requested attribution cannot be supported by available evidence."""
