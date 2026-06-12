"""Pure metric-computation functions.

Ported from the user's existing CDDBuilder (hart_core/dev_env) with no
behavioural changes to the maths: Rothfusz heat-index formula, cumulative
degree-day grid, and day-count grid. Added `required_variables()` which is
new for this package (it drives manifest-based ensemble membership).

All grid functions operate on a single model's data for a single year and
expect a `time` dimension to reduce over (see `standardise_dims`).
"""

import numpy as np
import xarray as xr

from enclim.exceptions import ConfigError

# --- Rothfusz heat index coefficients (NOAA), inputs in degF / %RH -------
_C0 = -42.379
_C1 = 2.04901523
_C2 = 10.14333127
_C3 = -0.22475541
_C4 = -6.83783e-3
_C5 = -5.481717e-2
_C6 = 1.22874e-3
_C7 = 8.5282e-4
_C8 = -1.99e-6

# Rothfusz formula is only valid above this temperature / humidity;
# below it the heat index is just the air temperature.
_T_THRESHOLD_F = 80.0
_RH_THRESHOLD = 40.0

_VALID_METRIC_TYPES = ('cdd', 'hi', 'hd')

# Variable(s) required from the CCKP daily collection for each metric type.
# 'hi' and 'cdd' with humidity=True additionally need 'hurs'.
_BASE_VARIABLE = {
    'hd': 'tasmax',
    'hi': 'tasmax',
    'cdd': 'tas',
}


def _format_threshold(threshold):
    """27.5 -> '27p5', 35 or 35.0 -> '35'."""
    if float(threshold).is_integer():
        return str(int(threshold))
    return str(threshold).replace('.', 'p')


def variable_name(metric: dict) -> str:
    """Output data-variable name for a metric, e.g. 'hd27p5', 'hi35', 'cdd_hum21'."""
    mtype = metric['type']
    thr = _format_threshold(metric['threshold'])
    if mtype == 'cdd' and metric.get('humidity', False):
        return f'cdd_hum{thr}'
    return f'{mtype}{thr}'


def required_variables(metric: dict) -> set:
    """Set of CCKP source variables needed to compute this metric."""
    mtype = metric['type']
    if mtype not in _VALID_METRIC_TYPES:
        raise ConfigError(
            f"Unknown metric type {mtype!r}; must be one of {_VALID_METRIC_TYPES}"
        )
    base = {_BASE_VARIABLE[mtype]}
    if mtype == 'hi' or (mtype == 'cdd' and metric.get('humidity', False)):
        base.add('hurs')
    return base


def validate_metrics(metrics) -> None:
    """Validate the `metrics` section of the config. Raises ConfigError listing
    every problem found, not just the first."""
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("'metrics' must be a non-empty list")

    errors = []
    for i, metric in enumerate(metrics):
        if not isinstance(metric, dict):
            errors.append(f"metrics[{i}] must be a mapping, got {type(metric).__name__}")
            continue

        mtype = metric.get('type')
        if mtype not in _VALID_METRIC_TYPES:
            errors.append(f"metrics[{i}].type must be one of {_VALID_METRIC_TYPES}, got {mtype!r}")

        if 'threshold' not in metric:
            errors.append(f"metrics[{i}] is missing required key 'threshold'")
        elif not isinstance(metric['threshold'], (int, float)) or isinstance(metric['threshold'], bool):
            errors.append(
                f"metrics[{i}].threshold must be numeric, got {type(metric['threshold']).__name__}"
            )

        if 'humidity' in metric:
            if mtype != 'cdd':
                errors.append(f"metrics[{i}].humidity is only valid for type 'cdd' (got type {mtype!r})")
            if not isinstance(metric['humidity'], bool):
                errors.append(f"metrics[{i}].humidity must be a boolean")

        extra = set(metric) - {'type', 'threshold', 'humidity'}
        if extra:
            errors.append(f"metrics[{i}] has unrecognised key(s): {sorted(extra)}")

    if errors:
        raise ConfigError("Invalid 'metrics' configuration:\n  " + "\n  ".join(errors))

    # Duplicate output variable names would silently overwrite each other
    # in the output Dataset - catch that explicitly.
    names = [variable_name(m) for m in metrics]
    seen = set()
    dupes = set()
    for name in names:
        if name in seen:
            dupes.add(name)
        seen.add(name)
    if dupes:
        raise ConfigError(f"Duplicate metric output variable name(s): {sorted(dupes)}")


def compute_cdd_grid(temp_c: xr.DataArray, threshold: float) -> xr.DataArray:
    """Cumulative degree-days above `threshold` (deg C), summed over time."""
    return xr.where(temp_c > threshold, temp_c - threshold, 0.0).sum(dim='time')


def compute_day_count_grid(temp_c: xr.DataArray, threshold: float) -> xr.DataArray:
    """Count of days where `temp_c` exceeds `threshold` (deg C), summed over time."""
    return xr.where(temp_c > threshold, 1, 0).sum(dim='time')


def compute_heat_index(t_f: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
    """NOAA Rothfusz heat index. Inputs in degF / %RH, output in degF.

    Below the validity threshold (T<=80F or RH<40%) the heat index is
    just the air temperature, per the NOAA definition.
    """
    hi_full = (
        _C0 + _C1 * t_f + _C2 * rh + _C3 * t_f * rh
        + _C4 * t_f ** 2 + _C5 * rh ** 2
        + _C6 * t_f ** 2 * rh + _C7 * t_f * rh ** 2
        + _C8 * t_f ** 2 * rh ** 2
    )
    use_full = (t_f > _T_THRESHOLD_F) & (rh >= _RH_THRESHOLD)
    return xr.where(use_full, hi_full, t_f)


# Common CF-ish dimension/coordinate name aliases seen across CMIP6-derived
# products. Anything not in this map is assumed to already be correct.
_DIM_ALIASES = {
    'latitude': 'lat',
    'longitude': 'lon',
}


def standardise_dims(da: xr.DataArray) -> xr.DataArray:
    """Rename lat/lon/time aliases, sort lat/lon ascending, and transpose to
    (time, lat, lon). Raises if `time` is missing after renaming, since every
    metric in this package reduces over time."""
    rename = {old: new for old, new in _DIM_ALIASES.items() if old in da.dims}
    if rename:
        da = da.rename(rename)

    if 'time' not in da.dims:
        raise ConfigError(
            f"Expected a 'time' dimension after standardising, got dims {da.dims}"
        )

    for dim in ('lat', 'lon'):
        if dim in da.coords and not np.all(np.diff(da[dim].values) > 0):
            da = da.sortby(dim)

    return da.transpose('time', 'lat', 'lon')
