class DeciShiftError(Exception):
    """Base exception for DeciShift."""


class ConfigurationError(DeciShiftError):
    """Raised when local configuration is invalid."""


class InsufficientEvidenceError(DeciShiftError):
    """Raised when requested analysis cannot be supported by available evidence."""


class ReproducibilityError(DeciShiftError):
    """Raised when strict reproducibility requirements are not met."""


class EvidenceIntegrityError(DeciShiftError):
    """Raised when a saved evidence bundle fails integrity verification."""


class ComponentExecutionError(DeciShiftError):
    """Raised when a user-supplied pipeline component fails during execution."""
