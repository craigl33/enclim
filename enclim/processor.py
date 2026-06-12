"""Per-model-year processing.

For one (dataset, year), loads each required source variable from S3
exactly once, then computes every configured metric that needs it. This
"load once, compute all metrics" pattern matters because each S3 read is a
~800-900MB file read - re-opening the same variable/year per metric would
multiply the dominant cost of an AWS run.
"""

import logging
import time

import xarray as xr

from enclim.metrics import (
    compute_cdd_grid,
    compute_day_count_grid,
    compute_heat_index,
    required_variables,
    variable_name,
)

logger = logging.getLogger(__name__)


def _to_fahrenheit(temp_c: xr.DataArray) -> xr.DataArray:
    return temp_c * 9.0 / 5.0 + 32.0


def _to_celsius(temp_f: xr.DataArray) -> xr.DataArray:
    return (temp_f - 32.0) * 5.0 / 9.0


class ModelYearProcessor:
    """Loads source variables for one (dataset, year) and computes metrics."""

    def __init__(self, config, loader):
        self._config = config
        self._loader = loader

    def process(self, dataset: str, year: int, metrics: list) -> dict:
        """Returns {output_variable_name: xr.DataArray} for each metric in `metrics`.

        Raises DataLoadError (propagated from the loader) if any required
        variable for this dataset/year cannot be loaded.
        """
        needed_vars = set()
        for metric in metrics:
            needed_vars |= required_variables(metric)

        loaded = {}
        for variable in needed_vars:
            start = time.perf_counter()
            loaded[variable] = self._loader.load_variable_year(variable, dataset, year)
            logger.debug("Loaded %s for %s/%s in %.2fs", variable, dataset, year, time.perf_counter() - start)

        results = {}
        for metric in metrics:
            start = time.perf_counter()
            results[variable_name(metric)] = self._compute_metric(metric, loaded)
            logger.debug(
                "Computed %s for %s/%s in %.2fs", variable_name(metric), dataset, year, time.perf_counter() - start
            )
        return results

    def _compute_metric(self, metric: dict, loaded: dict) -> xr.DataArray:
        mtype = metric['type']
        threshold = metric['threshold']

        if mtype == 'hd':
            return compute_day_count_grid(loaded['tasmax'], threshold)

        if mtype == 'hi':
            # Heat index threshold is treated in the same units (deg C) as
            # every other threshold in this config, so convert the Rothfusz
            # result (degF) back to degC before counting days.
            hi_f = compute_heat_index(_to_fahrenheit(loaded['tasmax']), loaded['hurs'])
            return compute_day_count_grid(_to_celsius(hi_f), threshold)

        if mtype == 'cdd':
            if metric.get('humidity', False):
                hi_f = compute_heat_index(_to_fahrenheit(loaded['tas']), loaded['hurs'])
                return compute_cdd_grid(_to_celsius(hi_f), threshold)
            return compute_cdd_grid(loaded['tas'], threshold)

        # Unreachable: metric types are validated by EnsembleConfig at load time.
        raise ValueError(f"Unknown metric type: {mtype!r}")
