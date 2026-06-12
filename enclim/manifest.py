"""S3-based manifest of available CMIP6 daily datasets per variable.

Discovers the full set of '{model}-{run}-{scenario}' dataset directories
available under each configured variable (analogous to NEX-GDDP's
index_v2.0_md5.txt), caches the result to JSON, and resolves per-metric
ensemble membership for a given scenario - e.g. the 'hi' metric needs the
intersection of tasmax-available and hurs-available datasets, which can
differ from the 'hd' metric's (tasmax-only) ensemble.
"""

import json
import logging
from pathlib import Path

import s3fs

from enclim.exceptions import ManifestError
from enclim.metrics import required_variables

logger = logging.getLogger(__name__)


class CCKPManifest:
    """Loads (or builds + caches) the list of available datasets per variable."""

    def __init__(self, config):
        self._config = config
        self._fs = s3fs.S3FileSystem(anon=config.anon)
        self._data = self._load()

    def _load(self) -> dict:
        cache_path = self._config.manifest_cache_path
        if not self._config.manifest_refresh and cache_path.exists():
            logger.info("Loading manifest cache from %s", cache_path)
            with open(cache_path) as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError as exc:
                    raise ManifestError(f"Manifest cache {cache_path} is not valid JSON: {exc}") from exc

        data = self._build()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Wrote manifest cache to %s", cache_path)
        return data

    def _build(self) -> dict:
        data = {}
        for variable in self._config.manifest_variables:
            prefix = f"{self._config.bucket}/data/{self._config.collection}/{variable}/"
            try:
                entries = self._fs.ls(prefix)
            except Exception as exc:
                raise ManifestError(f"Failed to list S3 prefix 's3://{prefix}': {exc}") from exc
            datasets = sorted(Path(e).name for e in entries)
            if not datasets:
                raise ManifestError(
                    f"No datasets found under 's3://{prefix}' - check source.bucket/collection "
                    f"and manifest.variables in the config"
                )
            data[variable] = datasets
            logger.info("Discovered %d datasets for variable %r", len(datasets), variable)
        return data

    def datasets_for_variable(self, variable: str) -> set:
        if variable not in self._data:
            raise ManifestError(
                f"Variable {variable!r} not in manifest (configured manifest.variables: {list(self._data)}). "
                f"Add it to manifest.variables and re-run with manifest.refresh: true."
            )
        return set(self._data[variable])

    def models_for_metric(self, metric: dict, scenario: str) -> list:
        """Datasets ('{model}-{run}-{scenario}') usable for `metric` under `scenario`.

        This is the intersection, across every variable `metric` requires, of
        datasets ending in '-{scenario}'. If `config.models` is an explicit
        list rather than 'auto', the result is further restricted to that list.
        May return an empty list - callers decide whether that's fatal.
        """
        candidates = None
        for variable in required_variables(metric):
            var_datasets = {
                d for d in self.datasets_for_variable(variable)
                if d.rpartition('-')[2] == scenario
            }
            candidates = var_datasets if candidates is None else candidates & var_datasets

        if self._config.models != 'auto':
            candidates &= set(self._config.models)

        return sorted(candidates)
