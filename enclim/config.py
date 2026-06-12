"""Configuration loading and validation.

Everything that controls a run (source location, manifest behaviour, model
selection, scenarios/years, metrics, output) lives in a single YAML file.
EnsembleConfig loads it once, validates it thoroughly (raising ConfigError
with *every* problem found, not just the first), and exposes typed
properties so the rest of the package never touches raw dict keys.
"""

import logging
from pathlib import Path

import yaml

from enclim.exceptions import ConfigError
from enclim.metrics import required_variables as _metric_required_variables
from enclim.metrics import validate_metrics

_REQUIRED_SECTIONS = ('source', 'manifest', 'models', 'scenarios', 'metrics', 'output', 'logging')
_REQUIRED_SOURCE_KEYS = ('bucket', 'collection', 'path_template', 'stat', 'anon')
_REQUIRED_MANIFEST_KEYS = ('cache_path', 'refresh', 'variables')
_REQUIRED_OUTPUT_KEYS = ('dir', 'filename_template')
_REQUIRED_LOGGING_KEYS = ('level',)
_VALID_LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')


class EnsembleConfig:
    """Loads and validates ensemble_config.yaml."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._raw = self._load_yaml(self.path)
        self._validate()

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        with open(path) as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ConfigError(f"Failed to parse YAML in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Top-level config in {path} must be a mapping, got {type(data).__name__}")
        return data

    def _validate(self) -> None:
        errors = []
        raw = self._raw

        for section in _REQUIRED_SECTIONS:
            if section not in raw:
                errors.append(f"Missing top-level section: '{section}'")
        if errors:
            raise ConfigError("Invalid config:\n  " + "\n  ".join(errors))

        source = raw['source']
        for key in _REQUIRED_SOURCE_KEYS:
            if key not in source:
                errors.append(f"Missing 'source.{key}'")
        if 'anon' in source and not isinstance(source['anon'], bool):
            errors.append("'source.anon' must be a boolean")

        manifest = raw['manifest']
        for key in _REQUIRED_MANIFEST_KEYS:
            if key not in manifest:
                errors.append(f"Missing 'manifest.{key}'")
        if 'variables' in manifest and (
            not isinstance(manifest['variables'], list) or not manifest['variables']
        ):
            errors.append("'manifest.variables' must be a non-empty list")
        if 'refresh' in manifest and not isinstance(manifest['refresh'], bool):
            errors.append("'manifest.refresh' must be a boolean")

        models = raw['models']
        if models != 'auto' and not isinstance(models, list):
            errors.append("'models' must be the string 'auto' or a list of dataset strings")

        scenarios = raw['scenarios']
        if not isinstance(scenarios, dict) or not scenarios:
            errors.append("'scenarios' must be a non-empty mapping of scenario name -> {year_start, year_end}")
        else:
            for name, yr in scenarios.items():
                if not isinstance(yr, dict) or 'year_start' not in yr or 'year_end' not in yr:
                    errors.append(f"scenarios.{name} must be a mapping with 'year_start' and 'year_end'")
                elif yr['year_start'] > yr['year_end']:
                    errors.append(f"scenarios.{name}: year_start ({yr['year_start']}) must be <= year_end ({yr['year_end']})")

        output = raw['output']
        for key in _REQUIRED_OUTPUT_KEYS:
            if key not in output:
                errors.append(f"Missing 'output.{key}'")

        logging_cfg = raw['logging']
        for key in _REQUIRED_LOGGING_KEYS:
            if key not in logging_cfg:
                errors.append(f"Missing 'logging.{key}'")
        if 'level' in logging_cfg and logging_cfg['level'] not in _VALID_LOG_LEVELS:
            errors.append(f"'logging.level' must be one of {_VALID_LOG_LEVELS}, got {logging_cfg['level']!r}")

        if errors:
            raise ConfigError("Invalid config:\n  " + "\n  ".join(errors))

        # metrics: validate shape/content (raises ConfigError with details)
        validate_metrics(raw['metrics'])

        # Cross-check: manifest.variables must cover everything the
        # configured metrics need, otherwise ensemble membership for
        # those metrics would silently resolve to an empty set.
        needed = set()
        for m in raw['metrics']:
            needed |= _metric_required_variables(m)
        missing = needed - set(manifest['variables'])
        if missing:
            raise ConfigError(
                f"'manifest.variables' ({manifest['variables']}) is missing variable(s) "
                f"required by configured metrics: {sorted(missing)}. Add them to manifest.variables."
            )

    # -- source ------------------------------------------------------
    @property
    def bucket(self) -> str:
        return self._raw['source']['bucket']

    @property
    def collection(self) -> str:
        return self._raw['source']['collection']

    @property
    def path_template(self) -> str:
        return self._raw['source']['path_template']

    @property
    def stat(self) -> str:
        return self._raw['source']['stat']

    @property
    def anon(self) -> bool:
        return self._raw['source']['anon']

    @property
    def chunks(self):
        """Optional dask chunking dict for xr.open_dataset, e.g. {'time': 50}.
        Returns None if not configured (no chunking)."""
        return self._raw['source'].get('chunks')

    # -- manifest ------------------------------------------------------
    @property
    def manifest_cache_path(self) -> Path:
        return Path(self._raw['manifest']['cache_path'])

    @property
    def manifest_refresh(self) -> bool:
        return self._raw['manifest']['refresh']

    @property
    def manifest_variables(self) -> list:
        return list(self._raw['manifest']['variables'])

    # -- models / scenarios / metrics -----------------------------------
    @property
    def models(self):
        """'auto' or an explicit list of '{model}-{run}-{scenario}' dataset strings."""
        return self._raw['models']

    @property
    def scenarios(self) -> dict:
        return self._raw['scenarios']

    @property
    def metrics(self) -> list:
        return self._raw['metrics']

    def scenario_years(self) -> list:
        """Expand `scenarios` into a flat list of (scenario, year) tuples,
        in config order, sorted by year within each scenario."""
        pairs = []
        for scenario, yr in self.scenarios.items():
            for year in range(yr['year_start'], yr['year_end'] + 1):
                pairs.append((scenario, year))
        return pairs

    def required_variables(self) -> set:
        """Union of source variables required across all configured metrics."""
        needed = set()
        for m in self.metrics:
            needed |= _metric_required_variables(m)
        return needed

    # -- output ------------------------------------------------------
    @property
    def output_dir(self) -> Path:
        return Path(self._raw['output']['dir'])

    @property
    def output_filename_template(self) -> str:
        return self._raw['output']['filename_template']

    # -- logging ------------------------------------------------------
    @property
    def log_level(self) -> int:
        return getattr(logging, self._raw['logging']['level'])
