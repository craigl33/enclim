"""Builds per-(scenario, year) ensemble-median metric grids across models."""

import logging

import xarray as xr

from enclim.exceptions import DataLoadError, EnsembleError
from enclim.io import CCKPDailyLoader
from enclim.manifest import CCKPManifest
from enclim.metrics import variable_name
from enclim.processor import ModelYearProcessor

logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """Resolves ensemble membership and computes the median metric grids."""

    def __init__(self, config):
        self._config = config
        self._manifest = CCKPManifest(config)
        self._loader = CCKPDailyLoader(config)
        self._processor = ModelYearProcessor(config, self._loader)

    def build(self, scenario: str, year: int) -> dict:
        """Returns {output_variable_name: xr.DataArray}, each the ensemble
        median across every model that successfully provided that metric for
        this scenario/year.

        Raises EnsembleError if no models are configured for a metric, or if
        every model fails to load for a metric.
        """
        metrics = self._config.metrics

        metric_datasets = {}
        for metric in metrics:
            name = variable_name(metric)
            datasets = self._manifest.models_for_metric(metric, scenario)
            if not datasets:
                raise EnsembleError(
                    f"No models available for metric '{name}' in scenario '{scenario}' - "
                    f"check manifest.variables, source.collection, and models in the config"
                )
            metric_datasets[name] = datasets
            logger.info("Metric '%s': %d candidate model(s) for scenario '%s'", name, len(datasets), scenario)

        all_datasets = sorted(set().union(*metric_datasets.values()))

        grids = {name: [] for name in metric_datasets}
        used_models = {name: [] for name in metric_datasets}

        for dataset in all_datasets:
            applicable = [m for m in metrics if dataset in metric_datasets[variable_name(m)]]
            try:
                results = self._processor.process(dataset, year, applicable)
            except DataLoadError as exc:
                logger.error("Skipping %s for %s/%s: %s", dataset, scenario, year, exc)
                continue

            for name, grid in results.items():
                grids[name].append(grid)
                used_models[name].append(dataset)

        ensemble = {}
        for name, members in grids.items():
            if not members:
                raise EnsembleError(
                    f"All models failed to load for metric '{name}' in scenario '{scenario}'/{year} "
                    f"- see preceding error log entries"
                )
            stacked = xr.concat(members, dim='model')
            median = stacked.median(dim='model')
            median.name = name
            median.attrs['n_models'] = len(members)
            median.attrs['models'] = ', '.join(used_models[name])
            ensemble[name] = median
            logger.info("Metric '%s': ensemble median across %d model(s)", name, len(members))

        return ensemble
