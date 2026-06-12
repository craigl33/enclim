"""Tests for enclim.config.EnsembleConfig - loading, validation, and the
typed properties the rest of the package relies on."""

import logging

import pytest

from conftest import base_config_dict, write_config
from enclim.config import EnsembleConfig
from enclim.exceptions import ConfigError


# -- file-level errors -------------------------------------------------

def test_config_file_not_found(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        EnsembleConfig(str(tmp_path / 'does_not_exist.yaml'))


def test_top_level_must_be_mapping(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        EnsembleConfig(str(path))


# -- structural validation -------------------------------------------------

def test_missing_top_level_section(tmp_path):
    cfg = base_config_dict(tmp_path)
    del cfg['logging']
    path = write_config(tmp_path, cfg)
    with pytest.raises(ConfigError, match=r"Missing top-level section: 'logging'"):
        EnsembleConfig(path)


def test_source_anon_must_be_bool(tmp_path):
    cfg = base_config_dict(tmp_path)
    cfg['source']['anon'] = 'true'
    path = write_config(tmp_path, cfg)
    with pytest.raises(ConfigError, match="'source.anon' must be a boolean"):
        EnsembleConfig(path)


def test_models_wrong_type(tmp_path):
    cfg = base_config_dict(tmp_path)
    cfg['models'] = 123
    path = write_config(tmp_path, cfg)
    with pytest.raises(ConfigError, match=r"'models' must be"):
        EnsembleConfig(path)


def test_year_start_after_year_end(tmp_path):
    cfg = base_config_dict(tmp_path)
    cfg['scenarios']['historical'] = {'year_start': 2020, 'year_end': 2014}
    path = write_config(tmp_path, cfg)
    with pytest.raises(ConfigError, match="must be <= year_end"):
        EnsembleConfig(path)


def test_invalid_log_level(tmp_path):
    cfg = base_config_dict(tmp_path)
    cfg['logging']['level'] = 'VERBOSE'
    path = write_config(tmp_path, cfg)
    with pytest.raises(ConfigError, match="'logging.level' must be one of"):
        EnsembleConfig(path)


# -- metrics validation (delegated to enclim.metrics.validate_metrics) ----

def test_invalid_metric_type(tmp_path):
    cfg = base_config_dict(tmp_path)
    cfg['metrics'] = [{'type': 'bogus', 'threshold': 1}]
    path = write_config(tmp_path, cfg)
    with pytest.raises(ConfigError, match="type must be one of"):
        EnsembleConfig(path)


def test_duplicate_metric_names(tmp_path):
    cfg = base_config_dict(tmp_path)
    cfg['metrics'] = [
        {'type': 'hd', 'threshold': 27.5},
        {'type': 'hd', 'threshold': 27.5},
    ]
    path = write_config(tmp_path, cfg)
    with pytest.raises(ConfigError, match="Duplicate metric output variable name"):
        EnsembleConfig(path)


# -- cross-check: manifest.variables vs configured metrics -----------------

def test_manifest_variables_missing_required_variable(tmp_path):
    cfg = base_config_dict(tmp_path)
    # 'hi' metric needs tasmax + hurs, but hurs is dropped here.
    cfg['manifest']['variables'] = ['tas', 'tasmax']
    path = write_config(tmp_path, cfg)
    with pytest.raises(ConfigError, match="missing variable"):
        EnsembleConfig(path)


# -- valid config: typed properties -------------------------------------

def test_valid_config_properties(tmp_path):
    cfg = base_config_dict(tmp_path)
    path = write_config(tmp_path, cfg)
    config = EnsembleConfig(path)

    assert config.bucket == 'wbg-cckp'
    assert config.collection == 'cmip6-daily-x0.25'
    assert config.stat == 'mean'
    assert config.anon is True
    assert config.chunks is None
    assert config.manifest_refresh is False
    assert config.manifest_variables == ['tas', 'tasmax', 'hurs']
    assert config.models == 'auto'
    assert config.scenario_years() == [('historical', 2014)]
    assert config.required_variables() == {'tasmax', 'hurs'}
    assert config.output_filename_template == 'cckp-ensemble-median_{scenario}_{year}.nc'
    assert config.log_level == logging.INFO


def test_scenario_years_expands_range(tmp_path):
    cfg = base_config_dict(tmp_path)
    cfg['scenarios'] = {'ssp245': {'year_start': 2030, 'year_end': 2032}}
    path = write_config(tmp_path, cfg)
    config = EnsembleConfig(path)

    assert config.scenario_years() == [
        ('ssp245', 2030), ('ssp245', 2031), ('ssp245', 2032),
    ]


def test_chunks_passthrough(tmp_path):
    cfg = base_config_dict(tmp_path)
    cfg['source']['chunks'] = {'time': 50}
    path = write_config(tmp_path, cfg)
    config = EnsembleConfig(path)

    assert config.chunks == {'time': 50}
