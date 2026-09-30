"""Typed exceptions. The pipeline fails loudly: these are raised, never swallowed."""


class FallsMLError(Exception):
    """Base class for all falls_ml errors."""


class ConfigError(FallsMLError):
    """Invalid, incomplete or contradictory configuration."""


class DatasetValidationError(FallsMLError):
    """The modelling dataset violates its declared schema or manifest."""

    def __init__(self, message: str, problems: list[str] | None = None):
        self.problems = list(problems or [])
        detail = "".join(f"\n  - {p}" for p in self.problems)
        super().__init__(f"{message}{detail}")


class LeakageError(FallsMLError):
    """An operation would let information cross a forbidden boundary (time, split, label)."""


class PreprocessingMismatchError(FallsMLError):
    """Preprocessing at inference differs from the preprocessing used at training."""


class BundleIntegrityError(FallsMLError):
    """A persisted model bundle is incomplete, tampered with, or incompatible."""


class DegenerateFitError(FallsMLError):
    """A fit is impossible for this particular data (e.g. separation, single outcome class after omissions).

    Raised loudly for the primary fit. Resampling analyses (bootstrap stability/optimism) record such
    replicates as failures and raise only when failures exceed their documented threshold.
    """


class NotFittedError(FallsMLError):
    """A component was used before being fitted."""
