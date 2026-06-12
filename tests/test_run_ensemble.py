"""Tests for enclim.run_ensemble._resolve_targets - maps CLI args /
AWS_BATCH_JOB_ARRAY_INDEX onto (scenario, year) targets."""

import pytest

from conftest import base_config_dict, write_config
from enclim.config import EnsembleConfig
from enclim.exceptions import ConfigError
from enclim.run_ensemble import _resolve_targets


@pytest.fixture
def multi_year_config(tmp_path, manifest_cache_path):
    cfg = base_config_dict(tmp_path)
    cfg['manifest']['cache_path'] = str(manifest_cache_path)
    cfg['scenarios'] = {'historical': {'year_start': 2012, 'year_end': 2014}}
    path = write_config(tmp_path, cfg)
    return EnsembleConfig(path)


def test_default_sweep_returns_all_scenario_years(multi_year_config):
    targets = _resolve_targets(multi_year_config, scenario=None, year=None)
    assert targets == [
        ('historical', 2012), ('historical', 2013), ('historical', 2014),
    ]


def test_explicit_scenario_and_year(multi_year_config):
    targets = _resolve_targets(multi_year_config, scenario='historical', year=2013)
    assert targets == [('historical', 2013)]


@pytest.mark.parametrize('scenario, year', [
    ('historical', None),
    (None, 2013),
])
def test_mismatched_scenario_year_raises(multi_year_config, scenario, year):
    with pytest.raises(ConfigError, match="must be given together"):
        _resolve_targets(multi_year_config, scenario=scenario, year=year)


def test_array_index_selects_single_target(multi_year_config, monkeypatch):
    monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '1')
    targets = _resolve_targets(multi_year_config, scenario=None, year=None)
    assert targets == [('historical', 2013)]


def test_array_index_out_of_range_raises(multi_year_config, monkeypatch):
    monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '99')
    with pytest.raises(ConfigError, match="out of range"):
        _resolve_targets(multi_year_config, scenario=None, year=None)


def test_array_index_non_integer_raises(multi_year_config, monkeypatch):
    monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', 'not-an-int')
    with pytest.raises(ConfigError, match="is not an integer"):
        _resolve_targets(multi_year_config, scenario=None, year=None)
