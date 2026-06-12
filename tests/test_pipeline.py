"""End-to-end (but offline) pipeline tests: ModelYearProcessor's
load-once-per-model-year behaviour, EnsembleBuilder's manifest-driven
membership + median computation, and NetCDFWriter's round trip.

No network access: CCKPManifest reads from a cached manifest JSON
(see conftest.config_path), and CCKPDailyLoader is replaced with
FakeLoader via monkeypatch.
"""

import numpy as np
import pytest
import xarray as xr

from conftest import FakeLoader, make_daily_grid
from enclim.config import EnsembleConfig
from enclim.ensemble import EnsembleBuilder
from enclim.exceptions import DataLoadError, EnsembleError
from enclim.processor import ModelYearProcessor
from enclim.writer import NetCDFWriter

HD_HI_METRICS = [
    {'type': 'hd', 'threshold': 27.5},
    {'type': 'hi', 'threshold': 35},
]


# -- ModelYearProcessor -------------------------------------------------

def test_model_year_processor_load_once_and_compute_all_metrics():
    tasmax = make_daily_grid([20.0, 30.0, 35.0])
    hurs = make_daily_grid([60.0, 60.0, 60.0])

    loader = FakeLoader({
        ('tasmax', 'modela-r1i1p1f1-historical'): tasmax,
        ('hurs', 'modela-r1i1p1f1-historical'): hurs,
    })
    # processor.process() never touches config, so None is fine here.
    processor = ModelYearProcessor(config=None, loader=loader)

    results = processor.process('modela-r1i1p1f1-historical', 2014, HD_HI_METRICS)

    # 'hd' needs tasmax, 'hi' needs tasmax + hurs -> 2 unique loads total,
    # not 4 (one per metric).
    assert sorted(loader.calls) == [
        ('hurs', 'modela-r1i1p1f1-historical', 2014),
        ('tasmax', 'modela-r1i1p1f1-historical', 2014),
    ]

    assert set(results) == {'hd27p5', 'hi35'}
    # tasmax > 27.5 on 2 of 3 days (30, 35).
    np.testing.assert_array_equal(results['hd27p5'].values, np.full((2, 2), 2))
    # At RH=60%, only 35C (95F) crosses the hi35 (35C) heat-index threshold;
    # 20C and 30C do not.
    np.testing.assert_array_equal(results['hi35'].values, np.full((2, 2), 1))


def test_model_year_processor_propagates_data_load_error():
    loader = FakeLoader(data={}, missing={'modelx-r1i1p1f1-historical'})
    processor = ModelYearProcessor(config=None, loader=loader)

    with pytest.raises(DataLoadError):
        processor.process('modelx-r1i1p1f1-historical', 2014, HD_HI_METRICS)


# -- EnsembleBuilder -------------------------------------------------

def test_ensemble_builder_end_to_end(config_path, monkeypatch):
    config = EnsembleConfig(config_path)

    # Manifest (see conftest.MANIFEST_DATASETS): hd27p5 -> A,B,C,D; hi35 -> A,C.
    data = {
        ('tasmax', 'modela-r1i1p1f1-historical'): make_daily_grid([20.0, 30.0, 35.0]),
        ('hurs', 'modela-r1i1p1f1-historical'): make_daily_grid([60.0, 60.0, 60.0]),
        ('tasmax', 'modelb-r1i1p1f1-historical'): make_daily_grid([10.0, 20.0, 28.0]),
        ('tasmax', 'modelc-r1i1p1f1-historical'): make_daily_grid([40.0, 10.0, 30.0]),
        ('hurs', 'modelc-r1i1p1f1-historical'): make_daily_grid([60.0, 60.0, 60.0]),
        ('tasmax', 'modeld-r1i1p1f1-historical'): make_daily_grid([27.0, 27.0, 27.0]),
    }
    loader = FakeLoader(data)
    monkeypatch.setattr('enclim.ensemble.CCKPDailyLoader', lambda cfg: loader)

    builder = EnsembleBuilder(config)
    ensemble = builder.build('historical', 2014)

    assert set(ensemble) == {'hd27p5', 'hi35'}

    # hd27p5 day-counts: A=2, B=1, C=2, D=0 -> median = 1.5
    hd = ensemble['hd27p5']
    assert hd.attrs['n_models'] == 4
    np.testing.assert_array_equal(hd.values, np.full((2, 2), 1.5))

    # hi35 day-counts: A=1, C=1 -> median = 1
    hi = ensemble['hi35']
    assert hi.attrs['n_models'] == 2
    np.testing.assert_array_equal(hi.values, np.full((2, 2), 1.0))

    # 'load once': tasmax for modela is needed by both hd27p5 and hi35, but
    # should only be fetched once.
    assert loader.calls.count(('tasmax', 'modela-r1i1p1f1-historical', 2014)) == 1


def test_ensemble_builder_raises_when_no_models_for_metric(config_path, monkeypatch):
    config = EnsembleConfig(config_path)
    loader = FakeLoader(data={})
    monkeypatch.setattr('enclim.ensemble.CCKPDailyLoader', lambda cfg: loader)

    builder = EnsembleBuilder(config)
    # The cached manifest only has '-historical' datasets; 'ssp245' has none.
    with pytest.raises(EnsembleError, match="No models available"):
        builder.build('ssp245', 2030)


def test_ensemble_builder_raises_when_all_models_fail(config_path, monkeypatch):
    config = EnsembleConfig(config_path)
    loader = FakeLoader(data={}, missing={
        'modela-r1i1p1f1-historical',
        'modelb-r1i1p1f1-historical',
        'modelc-r1i1p1f1-historical',
        'modeld-r1i1p1f1-historical',
    })
    monkeypatch.setattr('enclim.ensemble.CCKPDailyLoader', lambda cfg: loader)

    builder = EnsembleBuilder(config)
    with pytest.raises(EnsembleError, match="All models failed"):
        builder.build('historical', 2014)


# -- NetCDFWriter -------------------------------------------------

def test_netcdf_writer_round_trip(config_path):
    config = EnsembleConfig(config_path)

    grid = make_daily_grid([1.0, 2.0])
    median = grid.sum(dim='time')
    median.name = 'hd27p5'
    median.attrs['n_models'] = 3
    median.attrs['models'] = 'a, b, c'

    writer = NetCDFWriter(config)
    out_path = writer.write({'hd27p5': median}, 'historical', 2014)

    assert out_path.exists()

    with xr.open_dataset(out_path) as ds:
        assert 'hd27p5' in ds.data_vars
        np.testing.assert_array_equal(ds['hd27p5'].values, median.values)
        assert ds.attrs['scenario'] == 'historical'
        assert ds.attrs['year'] == 2014
        assert ds['hd27p5'].attrs['n_models'] == 3
