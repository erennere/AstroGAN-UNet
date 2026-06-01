from __future__ import annotations

import base64
import sys
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import starter


def _encode_raw(raw: bytes) -> str:
    return base64.urlsafe_b64encode(zlib.compress(raw)).decode('ascii').rstrip('=')


@pytest.mark.unit
def test_pack_and_read_tag_unmapped_and_marker_zero_path():
    packed = starter._pack_tag('custom-loss', starter._LOSS_CODE_MAP)
    assert packed.startswith(b'\x00')

    raw = b'\x00' + starter._pack_string('hello')
    value, cursor = starter._read_tag(raw, 0, {})
    assert value == 'hello'
    assert cursor == len(raw)


@pytest.mark.unit
def test_encode_payload_survey_list_and_scaling_null_path():
    token_d = starter._encode_alias_payload([1, ['UNKNOWN_SURVEY'], 0, [], 3, 7, 9])
    decoded_d = starter._decode_alias_payload(token_d)
    assert decoded_d[1] == ['UNKNOWN_SURVEY']

    token_m = starter._encode_alias_payload(['UNET', 'NOATTN', 'MSE', 'abc', 'null', '0p1', 'relu', 'sigmoid', 'relu', 'sigmoid'])
    decoded_m = starter._decode_alias_payload(token_m)
    assert decoded_m[4] == 'none'


@pytest.mark.unit
def test_decode_alias_payload_error_branches_for_short_and_trailing_and_kind():
    with pytest.raises(ValueError):
        starter._decode_alias_payload(_encode_raw(b'D\x01'))

    valid = starter._encode_alias_payload([1, ['UNKNOWN_SURVEY'], 0, [], 2, 3, 4])
    padded = valid + ('=' * (-len(valid) % 4))
    raw = zlib.decompress(base64.urlsafe_b64decode(padded.encode('ascii')))
    with pytest.raises(ValueError):
        starter._decode_alias_payload(_encode_raw(raw + b'X'))

    valid_m = starter._encode_alias_payload(['UNET', 'NOATTN', 'MSE', 'abc', 'none', '0p1', 'relu', 'sigmoid', 'relu', 'sigmoid'])
    padded_m = valid_m + ('=' * (-len(valid_m) % 4))
    raw_m = zlib.decompress(base64.urlsafe_b64decode(padded_m.encode('ascii')))
    with pytest.raises(ValueError):
        starter._decode_alias_payload(_encode_raw(raw_m + b'X'))

    with pytest.raises(ValueError):
        starter._decode_alias_payload(_encode_raw(b'Z\x01\x00'))


@pytest.mark.unit
def test_decode_data_alias_plain_header_and_trailing_bytes_errors():
    with pytest.raises(ValueError):
        starter._decode_data_alias_plain(_encode_raw(b'B\x01\x00'))

    token = starter._encode_data_alias_plain('SV_IR')
    padded = token + ('=' * (-len(token) % 4))
    raw = zlib.decompress(base64.urlsafe_b64decode(padded.encode('ascii')))
    with pytest.raises(ValueError):
        starter._decode_data_alias_plain(_encode_raw(raw + b'X'))


@pytest.mark.unit
def test_decode_data_signature_validation_and_canonicality_branches(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError):
        starter._decode_data_signature('')

    monkeypatch.setattr(starter, '_decode_alias_payload', lambda _token: ['x'])
    with pytest.raises(ValueError):
        starter._decode_data_signature('dummy')

    monkeypatch.setattr(starter, '_decode_alias_payload', lambda _token: [1, ['IR'], 0, [], 'bad', 3, 4])
    with pytest.raises(ValueError):
        starter._decode_data_signature('dummy')

    monkeypatch.setattr(starter, '_decode_alias_payload', lambda _token: [1, ['IR'], 0, [], 2, 3, 4])
    monkeypatch.setattr(starter, '_decode_data_alias_plain', lambda _token: 'DIFFERENT')
    with pytest.raises(ValueError):
        starter._decode_data_signature('dummy')

    monkeypatch.setattr(starter, '_decode_data_alias_plain', lambda _token: 'SV_IR')
    monkeypatch.setattr(starter, '_encode_alias_payload', lambda _payload: 'other')
    with pytest.raises(ValueError):
        starter._decode_data_signature('dummy')


@pytest.mark.unit
def test_decode_model_alias_validation_and_dropout_parse_paths(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError):
        starter._decode_model_alias('')

    monkeypatch.setattr(starter, '_decode_alias_payload', lambda _token: ['x'])
    with pytest.raises(ValueError):
        starter._decode_model_alias('dummy')

    monkeypatch.setattr(starter, '_decode_alias_payload', lambda _token: ['UNET', 'NOATTN', 'MSE', 'abc', 'none', 'abc', 'relu', 'sigmoid', 'relu', 'sigmoid'])
    monkeypatch.setattr(starter, '_encode_alias_payload', lambda _payload: 'dummy')
    monkeypatch.setattr(starter, '_decode_data_signature', lambda _hex: {'data_alias_enriched_hex': 'abc'})
    out = starter._decode_model_alias('dummy')
    assert out['dropout_rate'] == 'abc'

    monkeypatch.setattr(starter, '_encode_alias_payload', lambda _payload: 'other')
    with pytest.raises(ValueError):
        starter._decode_model_alias('dummy')


@pytest.mark.unit
def test_decode_models_dir_dropout_non_float_path(monkeypatch: pytest.MonkeyPatch):
    data_sig = starter._encode_data_signature(True, ['IR'], False, [], 3, 7, 9)
    models_dir = (
        f"/tmp/models/{data_sig['data_alias_enriched_hex']}/UNET/NOATTN/MSE/none/DOabc/"
        "ACTrelu/OUTsigmoid/DACTrelu/DOUTsigmoid"
    )
    out = starter._decode_models_dir(models_dir)
    assert out['dropout_rate'] == 'abc'


@pytest.mark.unit
def test_parse_helpers_and_formatter_and_runtime_resolvers_misc_paths():
    assert starter.parse_optional_float(None, 'x') is None
    assert starter.parse_optional_bool(None, 'x') is None
    assert starter.parse_optional_bool(True, 'x') is True

    formatted = starter._format_with_known_templates('{v!a}-{v!q}', {'v': 'x'})
    assert formatted.startswith("'x'-")
    assert formatted.endswith('{v!q}')

    assert starter._resolve_name_from_runtime_registry('definitely_unknown_symbol') == 'definitely_unknown_symbol'
    assert starter._resolve_output_activation_value(None) is None
    assert starter._resolve_output_activation_value('not_known_activation') == 'not_known_activation'
    assert starter._resolve_initializer_value(None) is None


@pytest.mark.unit
def test_resolve_initializer_and_data_strategy_passthrough(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(starter, '_resolve_name_from_runtime_registry', lambda value: value)
    assert starter._resolve_initializer_value('plain_initializer') == 'plain_initializer'
    assert starter._resolve_data_strategy_value('noise_fn', 123) == 123


@pytest.mark.unit
def test_finalize_metrics_section_use_custom_test_images_path_assignment():
    cfg = starter.load_config()
    cfg['metrics']['use_custom_test_images'] = True
    starter._finalize_metrics_section(cfg['metrics'], cfg)
    assert cfg['metrics']['data_kwargs']['kwargs_data']['metadata_filepath'] == cfg['create_dataset']['noisy_filtered_metadata_output_file']


@pytest.mark.unit
def test_validate_checkpoint_config_specific_validation_errors():
    cfg = {
        'training': {
            'checkpoint_filename_pattern': '',
            'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:03d}.keras'},
            'checkpoint_custom_epoch': None,
        },
        'data': {},
    }
    with pytest.raises(ValueError):
        starter._validate_and_sync_checkpoint_config(cfg)

    cfg2 = {
        'training': {
            'checkpoint_filename_pattern': '{prefix}_{epoch:03d}.keras',
            'checkpoint_restore_kwargs': {'filename_pattern': '{prefix}_{epoch:04d}.keras'},
            'checkpoint_custom_epoch': None,
        },
        'data': {},
    }
    with pytest.raises(ValueError):
        starter._validate_and_sync_checkpoint_config(cfg2)
