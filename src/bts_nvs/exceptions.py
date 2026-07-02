from __future__ import annotations


class BTSNVSError(Exception):
    """Base error for the BTS NVS pipeline."""


class DataValidationError(BTSNVSError):
    """Raised when scene data is missing, inconsistent, or unsupported."""


class ExternalCommandError(BTSNVSError):
    """Raised when an external training/rendering command fails."""
