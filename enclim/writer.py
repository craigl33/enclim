"""Writes ensemble-median metric grids to a single zlib-compressed NetCDF
file per scenario/year."""

import logging
from pathlib import Path

import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar

from enclim.exceptions import OutputError

logger = logging.getLogger(__name__)


class NetCDFWriter:
    """Writes one {metric: grid} dict to one NetCDF file, optionally
    uploading it to S3 afterwards (config.output.s3_uri)."""

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

        s3_uri = self._config.output_s3_uri
        if s3_uri:
            self._upload_to_s3(out_path, filename, s3_uri)

        return out_path

    def _upload_to_s3(self, local_path: Path, filename: str, s3_uri: str) -> None:
        """Uploads local_path to '{s3_uri}/{filename}' using the default AWS
        credential chain (~/.aws/credentials locally, or the IAM task role on
        AWS Batch). Raises OutputError on failure."""
        dest = f"{s3_uri.rstrip('/')}/{filename}"
        try:
            s3fs.S3FileSystem().put(str(local_path), dest)
        except Exception as exc:
            raise OutputError(f"Failed to upload {local_path} to {dest}: {exc}") from exc
        logger.info("Uploaded %s to %s", local_path, dest)
