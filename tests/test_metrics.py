"""Tests for the pure functions in enclim.metrics."""

import numpy as np
import pytest
import xarray as xr

from conftest import make_daily_grid
from enclim.exceptions import ConfigError
from enclim.metrics import (
    compute_cdd_grid,
    compute_day_count_grid,
    compute_heat_index,
    required_variables,
    standardise_dims,
    validate_metrics,
    variable_name,
)


# -- variable_name ----------------------------------------------------------

@pytest.mark.parametrize('metric, expected', [
    ({'type': 'hd', 'threshold': 27.5}, 'hd27p5'),
    ({'type': 'hi', 'threshold': 35}, 'hi35'),
    ({'type': 'hi', 'threshold': 35.0}, 'hi35'),
    ({'type': 'cdd', 'threshold': 21}, 'cdd21'),
    ({'type': 'cdd', 'threshold': 21, 'humidity': True}, 'cdd_hum21'),
    ({'type': 'cdd', 'threshold': 21, 'humidity': False}, 'cdd21'),
])
def test_variable_name(metric, expected):
    assert variable_name(metric) == expected


# -- required_variables -------------------------------------------------

def test_required_variables_hd():
    assert required_variables({'type': 'hd', 'threshold': 27.5}) == {'tasmax'}


def test_required_variables_hi():
    assert required_variables({'type': 'hi', 'threshold': 35}) == {'tasmax', 'hurs'}


def test_required_variables_cdd_without_humidity():
    assert required_variables({'type': 'cdd', 'threshold': 21}) == {'tas'}


def test_required_variables_cdd_with_humidity():
    assert required_variables({'type': 'cdd', 'threshold': 21, 'humidity': True}) == {'tas', 'hurs'}


def test_required_variables_unknown_type_raises():
    with pytest.raises(ConfigError):
        required_variables({'type': 'bogus', 'threshold': 1})


# -- validate_metrics -------------------------------------------------

def test_validate_metrics_accepts_valid_list():
    validate_metrics([
        {'type': 'hd', 'threshold': 27.5},
        {'type': 'hi', 'threshold': 35},
        {'type': 'cdd', 'threshold': 21, 'humidity': True},
    ])


@pytest.mark.parametrize('metrics, match', [
    ([], "non-empty list"),
    ("not-a-list", "non-empty list"),
    ([1], "must be a mapping"),
    ([{'type': 'bogus', 'threshold': 1}], "type must be one of"),
    ([{'type': 'hd'}], "missing required key 'threshold'"),
    ([{'type': 'hd', 'threshold': 'hot'}], "threshold must be numeric"),
    ([{'type': 'hd', 'threshold': True}], "threshold must be numeric"),
    ([{'type': 'hd', 'threshold': 27.5, 'humidity': True}], "only valid for type 'cdd'"),
    ([{'type': 'cdd', 'threshold': 21, 'humidity': 'yes'}], "humidity must be a boolean"),
    ([{'type': 'hd', 'threshold': 27.5, 'extra': 1}], "unrecognised key"),
    (
        [{'type': 'hd', 'threshold': 27.5}, {'type': 'hd', 'threshold': 27.5}],
        "Duplicate metric output variable name",
    ),
])
def test_validate_metrics_rejects(metrics, match):
    with pytest.raises(ConfigError, match=match):
        validate_metrics(metrics)


# -- grid functions -------------------------------------------------

def test_compute_day_count_grid():
    grid = make_daily_grid([10.0, 30.0, 20.0, 28.0])
    counts = compute_day_count_grid(grid, threshold=25.0)
    np.testing.assert_array_equal(counts.values, np.full((2, 2), 2))


def test_compute_cdd_grid():
    grid = make_daily_grid([10.0, 30.0, 20.0, 28.0])
    cdd = compute_cdd_grid(grid, threshold=25.0)
    # (30 - 25) + (28 - 25) = 8
    np.testing.assert_allclose(cdd.values, np.full((2, 2), 8.0))


# -- heat index -------------------------------------------------

def test_compute_heat_index_below_temperature_threshold_is_passthrough():
    # T <= 80F -> HI == T, regardless of RH.
    t_f = make_daily_grid([75.0])
    rh = make_daily_grid([60.0])
    hi = compute_heat_index(t_f, rh)
    np.testing.assert_allclose(hi.values, 75.0)


def test_compute_heat_index_below_humidity_threshold_is_passthrough():
    # RH < 40% -> HI == T, even if T > 80F.
    t_f = make_daily_grid([90.0])
    rh = make_daily_grid([30.0])
    hi = compute_heat_index(t_f, rh)
    np.testing.assert_allclose(hi.values, 90.0)


def test_compute_heat_index_rothfusz_formula():
    # Reference value for the (unadjusted) Rothfusz regression at
    # T=90F, RH=50% is ~94.6F.
    t_f = make_daily_grid([90.0])
    rh = make_daily_grid([50.0])
    hi = compute_heat_index(t_f, rh)
    np.testing.assert_allclose(hi.values, 94.5969412, rtol=1e-6)


# -- standardise_dims -------------------------------------------------

def test_standardise_dims_renames_sorts_and_transposes():
    data = np.zeros((2, 2, 2))
    da = xr.DataArray(
        data,
        dims=('latitude', 'longitude', 'time'),
        coords={'latitude': [1.0, 0.0], 'longitude': [0.0, 1.0], 'time': [0, 1]},
    )
    out = standardise_dims(da)
    assert out.dims == ('time', 'lat', 'lon')
    assert list(out['lat'].values) == [0.0, 1.0]


def test_standardise_dims_missing_time_raises():
    data = np.zeros((2, 2))
    da = xr.DataArray(data, dims=('lat', 'lon'), coords={'lat': [0.0, 1.0], 'lon': [0.0, 1.0]})
    with pytest.raises(ConfigError):
        standardise_dims(da)
