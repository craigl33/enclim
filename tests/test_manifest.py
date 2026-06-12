"""Tests for enclim.manifest.CCKPManifest - per-metric ensemble membership
resolved entirely from a cached manifest JSON (no S3 access)."""

import pytest

from conftest import base_config_dict, write_config
from enclim.config import EnsembleConfig
from enclim.exceptions import ManifestError
from enclim.manifest import CCKPManifest

HD_METRIC = {'type': 'hd', 'threshold': 27.5}
HI_METRIC = {'type': 'hi', 'threshold': 35}


def test_models_for_metric_hd_uses_tasmax_only(config_path):
    config = EnsembleConfig(config_path)
    manifest = CCKPManifest(config)

    assert manifest.models_for_metric(HD_METRIC, 'historical') == [
        'modela-r1i1p1f1-historical',
        'modelb-r1i1p1f1-historical',
        'modelc-r1i1p1f1-historical',
        'modeld-r1i1p1f1-historical',
    ]


def test_models_for_metric_hi_intersects_tasmax_and_hurs(config_path):
    config = EnsembleConfig(config_path)
    manifest = CCKPManifest(config)

    assert manifest.models_for_metric(HI_METRIC, 'historical') == [
        'modela-r1i1p1f1-historical',
        'modelc-r1i1p1f1-historical',
    ]


def test_models_for_metric_filters_by_scenario(tmp_path, manifest_cache_path):
    cfg = base_config_dict(tmp_path)
    cfg['manifest']['cache_path'] = str(manifest_cache_path)
    cfg['scenarios'] = {'ssp245': {'year_start': 2030, 'year_end': 2030}}
    path = write_config(tmp_path, cfg)
    config = EnsembleConfig(path)
    manifest = CCKPManifest(config)

    # The cached manifest only contains '-historical' datasets, so nothing
    # matches scenario 'ssp245'.
    assert manifest.models_for_metric(HD_METRIC, 'ssp245') == []


def test_explicit_models_list_restricts_candidates(tmp_path, manifest_cache_path):
    cfg = base_config_dict(tmp_path)
    cfg['manifest']['cache_path'] = str(manifest_cache_path)
    cfg['models'] = [
        'modela-r1i1p1f1-historical',
        'modelb-r1i1p1f1-historical',
    ]
    path = write_config(tmp_path, cfg)
    config = EnsembleConfig(path)
    manifest = CCKPManifest(config)

    assert manifest.models_for_metric(HD_METRIC, 'historical') == [
        'modela-r1i1p1f1-historical',
        'modelb-r1i1p1f1-historical',
    ]
    # 'hi' also needs hurs; only modela has it among the explicit list.
    assert manifest.models_for_metric(HI_METRIC, 'historical') == [
        'modela-r1i1p1f1-historical',
    ]


def test_datasets_for_unconfigured_variable_raises(config_path):
    config = EnsembleConfig(config_path)
    manifest = CCKPManifest(config)

    with pytest.raises(ManifestError, match="not in manifest"):
        manifest.datasets_for_variable('pr')


def test_invalid_cache_json_raises_manifest_error(tmp_path):
    cache_path = tmp_path / 'manifest.json'
    cache_path.write_text("{not valid json")

    cfg = base_config_dict(tmp_path)
    cfg['manifest']['cache_path'] = str(cache_path)
    path = write_config(tmp_path, cfg)
    config = EnsembleConfig(path)

    with pytest.raises(ManifestError, match="not valid JSON"):
        CCKPManifest(config)
