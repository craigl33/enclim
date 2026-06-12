"""Shared fixtures for the enclim test suite."""

import json

import numpy as np
import pytest
import xarray as xr
import yaml

from enclim.exceptions import DataLoadError

# A small lat/lon grid shared by every fake DataArray.
_LAT = [0.0, 1.0]
_LON = [0.0, 1.0]


def make_daily_grid(daily_values):
    """Build a (time, lat, lon) DataArray where every cell on a given day
    has the same value (taken from `daily_values`)."""
    data = np.array(daily_values, dtype=float)[:, None, None] * np.ones((1, len(_LAT), len(_LON)))
    return xr.DataArray(
        data,
        dims=('time', 'lat', 'lon'),
        coords={'time': np.arange(len(daily_values)), 'lat': _LAT, 'lon': _LON},
    )


class FakeLoader:
    """Stand-in for CCKPDailyLoader. Records every call so tests can assert
    on the 'load once per model-year' behaviour, and raises DataLoadError
    (never returns a default) for datasets in `missing`."""

    def __init__(self, data, missing=None):
        self.data = data
        self.missing = missing or set()
        self.calls = []

    def load_variable_year(self, variable, dataset, year):
        self.calls.append((variable, dataset, year))
        if dataset in self.missing:
            raise DataLoadError(f"FakeLoader: no data for {dataset}/{year}")
        try:
            return self.data[(variable, dataset)]
        except KeyError:
            raise DataLoadError(f"FakeLoader: no {variable!r} for {dataset!r}")


# -- config ---------------------------------------------------------------

def base_config_dict(tmp_path):
    """A minimal, valid config dict. manifest.cache_path and output.dir are
    pointed at tmp_path so tests don't touch the real filesystem layout."""
    return {
        'source': {
            'bucket': 'wbg-cckp',
            'collection': 'cmip6-daily-x0.25',
            'path_template': (
                'data/{collection}/{variable}/{dataset}/'
                'timeseries-{variable}-daily-{stat}_{collection}_{dataset}_timeseries_{stat}_{year}.nc'
            ),
            'stat': 'mean',
            'anon': True,
        },
        'manifest': {
            'cache_path': str(tmp_path / 'manifest.json'),
            'refresh': False,
            'variables': ['tas', 'tasmax', 'hurs'],
        },
        'models': 'auto',
        'scenarios': {
            'historical': {'year_start': 2014, 'year_end': 2014},
        },
        'metrics': [
            {'type': 'hd', 'threshold': 27.5},
            {'type': 'hi', 'threshold': 35},
        ],
        'output': {
            'dir': str(tmp_path / 'output'),
            'filename_template': 'cckp-ensemble-median_{scenario}_{year}.nc',
        },
        'logging': {'level': 'INFO'},
    }


def write_config(tmp_path, config_dict, name='config.yaml'):
    """Write `config_dict` to `tmp_path/name` and return its path as a str."""
    path = tmp_path / name
    with open(path, 'w') as f:
        yaml.safe_dump(config_dict, f)
    return str(path)


# -- manifest ---------------------------------------------------------------

# 4 datasets have tasmax, only A and C also have hurs -> hd27p5 sees
# 4 models, hi35 sees 2.
MANIFEST_DATASETS = {
    'tas': [
        'modela-r1i1p1f1-historical',
        'modelb-r1i1p1f1-historical',
        'modelc-r1i1p1f1-historical',
        'modeld-r1i1p1f1-historical',
    ],
    'tasmax': [
        'modela-r1i1p1f1-historical',
        'modelb-r1i1p1f1-historical',
        'modelc-r1i1p1f1-historical',
        'modeld-r1i1p1f1-historical',
    ],
    'hurs': [
        'modela-r1i1p1f1-historical',
        'modelc-r1i1p1f1-historical',
    ],
}


@pytest.fixture
def manifest_cache_path(tmp_path):
    """Write MANIFEST_DATASETS to a manifest cache JSON and return its path."""
    path = tmp_path / 'manifest.json'
    with open(path, 'w') as f:
        json.dump(MANIFEST_DATASETS, f)
    return path


@pytest.fixture
def config_path(tmp_path, manifest_cache_path):
    """A valid config.yaml whose manifest.cache_path points at
    manifest_cache_path (refresh: false), so CCKPManifest never touches S3."""
    cfg = base_config_dict(tmp_path)
    cfg['manifest']['cache_path'] = str(manifest_cache_path)
    return write_config(tmp_path, cfg)
