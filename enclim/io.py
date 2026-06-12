"""Streaming access to CCKP CMIP6 daily NetCDF files on S3.

Files are opened via s3fs + the h5netcdf engine, which reads byte ranges
on demand rather than downloading the whole (~800-900MB) file. Optional
dask chunking (config.source.chunks) keeps peak memory bounded when
processing many models.
"""

import logging
import time

import s3fs
import xarray as xr

from enclim.exceptions import DataLoadError
from enclim.metrics import standardise_dims

logger = logging.getLogger(__name__)


class CCKPDailyLoader:
    """Builds S3 paths from config.source.path_template and opens them lazily."""

    def __init__(self, config):
        self._config = config
        self._fs = s3fs.S3FileSystem(anon=config.anon)

    def _s3_key(self, variable: str, dataset: str, year: int) -> str:
        return self._config.path_template.format(
            collection=self._config.collection,
            variable=variable,
            dataset=dataset,
            stat=self._config.stat,
            year=year,
        )

    def load_variable_year(self, variable: str, dataset: str, year: int) -> xr.DataArray:
        """Open `variable` for `dataset`/`year`.

        Returns a standardised DataArray with dims (time, lat, lon).
        Raises DataLoadError if the file is missing, unreadable, or doesn't
        contain `variable` - never returns a default/empty value.
        """
        key = self._s3_key(variable, dataset, year)
        url = f"s3://{self._config.bucket}/{key}"

        start = time.perf_counter()

        try:
            f = self._fs.open(url, 'rb')
        except FileNotFoundError as exc:
            raise DataLoadError(f"Source file not found: {url}") from exc
        except Exception as exc:
            raise DataLoadError(f"Failed to open {url}: {exc}") from exc

        try:
            ds = xr.open_dataset(f, engine='h5netcdf', chunks=self._config.chunks)
        except Exception as exc:
            raise DataLoadError(f"Failed to read {url} as NetCDF: {exc}") from exc

        data_var = self._config.variable_template.format(variable=variable, stat=self._config.stat)
        if data_var not in ds.data_vars:
            raise DataLoadError(
                f"Data variable {data_var!r} (for {variable!r}) not found in {url}; "
                f"available: {list(ds.data_vars)}"
            )

        elapsed = time.perf_counter() - start
        logger.debug("Opened %s in %.2fs (variable=%s, data_var=%s)", url, elapsed, variable, data_var)
        return standardise_dims(ds[data_var])
