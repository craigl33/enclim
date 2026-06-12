"""Custom exception types for enclim.

Using distinct exception types (rather than bare Exception / generic
ValueError everywhere) lets callers and logs distinguish configuration
problems from data problems from manifest problems at a glance.
"""


class ConfigError(Exception):
    """Raised when ensemble_config.yaml is missing, malformed, or internally inconsistent."""


class ManifestError(Exception):
    """Raised when the S3 model/run/scenario manifest cannot be built or read."""


class DataLoadError(Exception):
    """Raised when a source NetCDF file cannot be opened or is missing an expected variable."""


class EnsembleError(Exception):
    """Raised when an ensemble cannot be computed (e.g. no models available for a metric)."""
