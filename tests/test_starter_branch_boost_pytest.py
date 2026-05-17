"""Branch-focused coverage boosts for starter.py helpers."""
from __future__ import annotations

import base64
import json
import sys
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import starter


@pytest.mark.unit
def test_pack_and_read_u16_error_paths():
    with pytest.raises(ValueError):
        starter._pack_u16(-1)
    with pytest.raises(ValueError):
        starter._pack_u16(70000)
    with pytest.raises(ValueError):
        starter._read_u16(b'\x00', 0)


@pytest.mark.unit
def test_read_string_and_list_error_paths():
    with pytest.raises(ValueError):
        starter._read_string(b'\x00\x05ab', 0)
    with pytest.raises(ValueError):
        starter._read_string_list(b'', 0)


@pytest.mark.unit
def test_pack_string_list_too_many_items():
    values = [str(i) for i in range(256)]
    with pytest.raises(ValueError):
        starter._pack_string_list(values)


@pytest.mark.unit
def test_read_tag_error_paths():
    with pytest.raises(ValueError):
        starter._read_tag(b'', 0, {})
    with pytest.raises(ValueError):
        starter._read_tag(b'\x01', 0, {})
    with pytest.raises(ValueError):
        starter._read_tag(b'\x01\x7f', 0, {})
    with pytest.raises(ValueError):
        starter._read_tag(b'\x02', 0, {})


@pytest.mark.unit
def test_encode_alias_payload_validation_errors():
    with pytest.raises(ValueError):
        starter._encode_alias_payload('not-list')
    with pytest.raises(ValueError):
        starter._encode_alias_payload([1, 2, 3])


@pytest.mark.unit
def test_decode_alias_payload_validation_errors():
    with pytest.raises(ValueError):
        starter._decode_alias_payload('')
    with pytest.raises(ValueError):
        starter._decode_alias_payload('not_base64')

    raw = b'X' + b'\x02' + b'\x00'
    token = base64.urlsafe_b64encode(zlib.compress(raw)).decode('ascii').rstrip('=')
    with pytest.raises(ValueError):
        starter._decode_alias_payload(token)


@pytest.mark.unit
def test_decode_data_alias_plain_validation_errors():
    with pytest.raises(ValueError):
        starter._decode_data_alias_plain(123)
    with pytest.raises(ValueError):
        starter._decode_data_alias_plain('@@bad@@')


@pytest.mark.unit
def test_decode_models_dir_invalid_input():
    with pytest.raises(ValueError):
        starter._decode_models_dir('')
    with pytest.raises(ValueError):
        starter._decode_models_dir('invalid/path')


@pytest.mark.unit
def test_optional_parse_helpers_error_paths():
    with pytest.raises(ValueError):
        starter._parse_optional_int('x', 'nsigma')
    with pytest.raises(ValueError):
        starter._parse_optional_float('x', 'dropout_rate')
    with pytest.raises(ValueError):
        starter._parse_optional_bool('maybe', 'attention')


@pytest.mark.unit
def test_format_with_known_templates_bad_format_falls_back_token():
    result = starter._format_with_known_templates('{name:badformat}', {'name': object()})
    assert result == '{name:badformat}'


@pytest.mark.unit
def test_resolve_helpers_raise_for_unknown_values():
    with pytest.raises(ValueError):
        starter.resolve_generator_loss('x', {'a': 1})
    with pytest.raises(ValueError):
        starter.resolve_gan_loss('x', {'a': 1})
    with pytest.raises(ValueError):
        starter.resolve_data_generator('x', {'a': 1})


@pytest.mark.unit
def test_auto_resolve_training_entries_passthrough_on_non_dict():
    assert starter._auto_resolve_training_entries('x') == 'x'


@pytest.mark.unit
def test_parse_config_overrides_args_object_branch():
    class Args:
        def __init__(self):
            self.nsigma = '3'
            self.footprint_radius = '9'
            self.npixels = '11'
            self.model_type = 'gan'
            self.attention = 'true'
            self.scaling = 'min_max'
            self.loss_name = 'mae'
            self.dropout_rate = '0.1'
            self.output_activation = 'sigmoid'
            self.kernel_initializer = 'he_normal'
            self.activation_name = 'relu'
            self.discriminator_activation = 'leaky_relu'
            self.discriminator_output_activation = 'sigmoid'
            self.filter_surveys = 'false'
            self.filter_by_last_name = 'true'
            self.last_name_filter_value = 'a,b'

    parsed = starter.parse_config_overrides(args=Args())
    assert parsed['nsigma'] == 3
    assert parsed['attention'] is True
    assert parsed['filter_by_last_name'] is True
    assert parsed['last_name_filter_value'] == ['a', 'b']


@pytest.mark.unit
def test_checkpoint_binding_validation_errors(monkeypatch: pytest.MonkeyPatch):
    cfg = {
        'training': {
            'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras',
            'checkpoint_restore_kwargs': 'bad',
            'checkpoint_custom_epoch': None,
        },
        'data': {},
        'runtime': {},
    }
    with pytest.raises(TypeError):
        starter._validate_and_sync_checkpoint_config(cfg)

    cfg['training']['checkpoint_restore_kwargs'] = {}
    with pytest.raises(ValueError):
        starter._validate_and_sync_checkpoint_config(cfg)


@pytest.mark.unit
def test_bind_nested_shared_value_rejects_non_mapping():
    cfg = {'section': {'sub': 'bad'}, 'runtime': {}}
    with pytest.raises(TypeError):
        starter._bind_nested_shared_value(
            cfg,
            'section',
            'sub',
            'k',
            1,
            group_name='paths',
            policy={'prefer_explicit_values': True},
            source_path='x',
        )


@pytest.mark.unit
def test_get_shared_binding_policy_non_dict_defaults():
    cfg = {'shared_bindings': 'bad'}
    policy = starter._get_shared_binding_policy(cfg)
    assert policy['prefer_explicit_values'] is True
