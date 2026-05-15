from pathlib import Path

import pytest
import yaml

from starter import _apply_shared_runtime_bindings, _decode_models_dir, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_load_config_contains_expected_top_level_keys():
    cfg = load_config()

    for key in ('training', 'network', 'discriminator', 'data', 'paths', 'evaluation', 'visualization'):
        assert key in cfg


@pytest.mark.unit
def test_model_type_override_sets_use_gan_true():
    cfg = load_config(model_type='gan')

    assert cfg['training']['use_gan'] is True


@pytest.mark.unit
def test_scaling_override_propagates_to_paths_and_runtime_bindings():
    cfg = load_config(scaling='z_scale')

    assert cfg['training']['scaling'] == 'z_scale'
    assert cfg['evaluation']['scaling'] == 'z_scale'
    assert cfg['visualization']['prepare_images']['scaling'] == 'z_scale'
    assert '/z_scale/' in cfg['paths']['models_dir']
    assert 'z_scale' in cfg['training']['training_metrics_csv_path']


@pytest.mark.unit
def test_data_alias_is_deterministic_for_same_inputs():
    cfg_a = load_config(nsigma=3, footprint_radius=7, npixels=9, filter_surveys=True, filter_by_last_name=True, last_name_filter_value=['FABER'])
    cfg_b = load_config(nsigma=3, footprint_radius=7, npixels=9, filter_surveys=True, filter_by_last_name=True, last_name_filter_value=['FABER'])

    assert _decode_models_dir(cfg_a['paths']['models_dir'])['data_alias_enriched_hex'] == _decode_models_dir(cfg_b['paths']['models_dir'])['data_alias_enriched_hex']


@pytest.mark.unit
def test_data_alias_changes_when_any_input_changes():
    cfg_a = load_config(nsigma=3, footprint_radius=7, npixels=9, filter_surveys=True, filter_by_last_name=False)
    cfg_b = load_config(nsigma=4, footprint_radius=7, npixels=9, filter_surveys=True, filter_by_last_name=False)

    assert _decode_models_dir(cfg_a['paths']['models_dir'])['data_alias_enriched_hex'] != _decode_models_dir(cfg_b['paths']['models_dir'])['data_alias_enriched_hex']


@pytest.mark.unit
def test_model_alias_is_deterministic_for_same_model_inputs():
    cfg_a = load_config(model_type='unet', attention=False, scaling='min_max', loss_name='MeanAbsoluteError', output_activation='sigmoid', activation_name='ReLU')
    cfg_b = load_config(model_type='unet', attention=False, scaling='min_max', loss_name='MeanAbsoluteError', output_activation='sigmoid', activation_name='ReLU')

    assert _decode_models_dir(cfg_a['paths']['models_dir'])['model_alias_hex'] == _decode_models_dir(cfg_b['paths']['models_dir'])['model_alias_hex']


@pytest.mark.unit
def test_apply_shared_runtime_bindings_propagates_scaling_to_downstream_sections():
    cfg = load_config(scaling='log_min_max')
    cfg['evaluation']['scaling'] = None
    cfg['visualization']['prepare_images']['scaling'] = None

    _apply_shared_runtime_bindings(cfg)

    assert cfg['evaluation']['scaling'] == 'log_min_max'
    assert cfg['visualization']['prepare_images']['scaling'] == 'log_min_max'


@pytest.mark.unit
def test_load_config_raises_clear_exception_for_invalid_top_level_shape(tmp_path: Path):
    config_path = tmp_path / 'invalid.yaml'
    config_path.write_text('- not-a-mapping\n', encoding='utf-8')

    with pytest.raises(TypeError, match='top-level mapping'):
        load_config(config_path=config_path)


@pytest.mark.unit
def test_load_config_raises_clear_exception_for_missing_checkpoint_filename_pattern(tmp_path: Path):
    with open(REPO_ROOT / 'config.yaml', 'r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    del cfg['training']['checkpoint_restore_kwargs']['filename_pattern']
    config_path = tmp_path / 'broken.yaml'
    config_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')

    with pytest.raises(ValueError, match='filename_pattern'):
        load_config(config_path=config_path)
