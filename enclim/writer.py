"""Writes ensemble-median metric grids to a single zlib-compressed NetCDF
file per scenario/year."""

import logging
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

logger = logging.getLogger(__name__)


class NetCDFWriter:
    """Writes one {metric: grid} dict to one NetCDF file."""

    def __init__(self, config):
        self._config = config

    def write(self, ensemble: dict, scenario: str, year: int) -> Path:
        ds = xr.Dataset(ensemble)
        ds.attrs['scenario'] = scenario
        ds.attrs['year'] = year
        ds.attrs['source'] = f"s3://{self._config.bucket}/data/{self._config.collection}"

        out_dir = self._config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = self._config.output_filename_template.format(scenario=scenario, year=year)
        out_path = out_dir / filename

        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
        with ProgressBar():
            ds.to_netcdf(out_path, encoding=encoding)
        logger.info("Wrote %s", out_path)
        return out_path
