"""Minimal config loader for config.yaml."""

import copy
import inspect
import hashlib
import argparse
import base64
import importlib
import os
import re
import string
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
import yaml

from src.config_parsing import (
    parse_optional_bool,
    parse_optional_float,
    parse_optional_int,
    parse_optional_str,
)

CONFIG_PATH = Path(__file__).resolve().with_name('config.yaml')
RUNTIME_DATA_SCENARIO_MAP = {}

_KNOWN_SURVEYS = [
    'IR',
    'IR-FIX',
    'IR-UVIS-CENTER',
    'IR-UVIS-FIX',
    'GRISM1024',
    'IR-UVIS',
    'G141-REF',
    'G102-REF',
]

_LOSS_CODE_MAP = {
    'BCE': 1,
    'MSE': 2,
    'MAE': 3,
    'SSIM': 4,
    'SIMAE': 5,
    'LOGCOSH': 6,
}
_SCALING_CODE_MAP = {
    'none': 0,
    'z_scale': 1,
    'min_max': 2,
    'log_min_max': 3,
}
_ACTIVATION_CODE_MAP = {
    'none': 0,
    'relu': 1,
    'leakyrelu': 2,
    'sigmoid': 3,
    'tanh': 4,
}

_RUNTIME_IMPORT_SPECS = {
    'data_augment': ('src.training.new_train', 'data_augment'),
    'data_augment_pluggable': ('src.training.new_train', 'data_augment_pluggable'),
    'log_cosh_loss': ('src.training.utils', 'log_cosh_loss'),
    'scale_invariant_mae': ('src.training.utils', 'scale_invariant_mae'),
    'ssim_loss': ('src.training.utils', 'ssim_loss'),
    '_simulated_image_from_exposure': ('src.training.math_helpers', '_simulated_image_from_exposure'),
    '_simulated_image_from_poisson': ('src.training.math_helpers', '_simulated_image_from_poisson'),
    'wrap_extract_sources': ('src.evaluation.metrics', 'wrap_extract_sources'),
    'detect_sources_in_image': ('src.evaluation.metrics', 'detect_sources_in_image'),
}

# Function order (top -> bottom):
# 1) Override specification and parsing primitives
# 2) Alias encoding/decoding utilities
# 3) Template expansion and runtime symbol resolution
# 4) Override application and template-context construction
# 5) Runtime-source extraction and section-level value resolution
# 6) Section finalization and public loading API


class ConfigResolutionError(KeyError):
    """Raised when config references or derived sections cannot be resolved."""


@dataclass(frozen=True)
class _OverrideSpec:
    key: str
    flag: str
    positional_index: int
    coerce: Callable[[Any], Any]
    default_getter: Callable[[dict[str, Any], dict[str, Any]], Any]
    apply: Callable[[dict[str, Any], Any], None]


# ---------------------------------------------------------------------------
# 1) Override specification and parsing primitives
# ---------------------------------------------------------------------------


def _config_path_getter(*path_parts):
    def getter(config_data, _explicit_overrides):
        current = config_data
        for part in path_parts:
            current = current[part]
        return current

    return getter


def _config_path_setter(*path_parts):
    def setter(config_data, value):
        current = config_data
        for part in path_parts[:-1]:
            current = current[part]
        current[path_parts[-1]] = value

    return setter


def _coerce_last_name_filter_value(value):
    parsed = parse_optional_str(value)
    if parsed is None:
        return None
    return [name.strip() for name in parsed.split(',') if name.strip()]


def _default_model_type(config_data, _explicit_overrides):
    return 'gan' if config_data['new_train']['training']['use_gan'] else 'unet'


def _default_loss_name(config_data, explicit_overrides):
    model_type = explicit_overrides.get('model_type')
    if model_type is None:
        model_type = _default_model_type(config_data, explicit_overrides)
    if str(model_type).lower() == 'gan':
        return config_data['new_train']['gan']['loss_fn']
    return config_data['new_train']['training']['g_loss_fn']


def _apply_model_type_override(config_data, value):
    config_data['new_train']['training']['use_gan'] = str(value).lower() == 'gan'


_OVERRIDE_SPECS = (
    _OverrideSpec(
        key='nsigma',
        flag='--nsigma',
        positional_index=0,
        coerce=lambda value: parse_optional_int(value, 'nsigma'),
        default_getter=_config_path_getter('create_dataset', 'nsigma'),
        apply=_config_path_setter('create_dataset', 'nsigma'),
    ),
    _OverrideSpec(
        key='footprint_radius',
        flag='--footprint-radius',
        positional_index=1,
        coerce=lambda value: parse_optional_int(value, 'footprint_radius'),
        default_getter=_config_path_getter('create_dataset', 'footprint_radius'),
        apply=_config_path_setter('create_dataset', 'footprint_radius'),
    ),
    _OverrideSpec(
        key='npixels',
        flag='--npixels',
        positional_index=2,
        coerce=lambda value: parse_optional_int(value, 'npixels'),
        default_getter=_config_path_getter('create_dataset', 'npixels'),
        apply=_config_path_setter('create_dataset', 'npixels'),
    ),
    _OverrideSpec(
        key='model_type',
        flag='--model-type',
        positional_index=3,
        coerce=parse_optional_str,
        default_getter=_default_model_type,
        apply=_apply_model_type_override,
    ),
    _OverrideSpec(
        key='attention',
        flag='--attention',
        positional_index=4,
        coerce=lambda value: parse_optional_bool(value, 'attention'),
        default_getter=_config_path_getter('new_train', 'network', 'attention'),
        apply=_config_path_setter('new_train', 'network', 'attention'),
    ),
    _OverrideSpec(
        key='scaling',
        flag='--scaling',
        positional_index=5,
        coerce=parse_optional_str,
        default_getter=_config_path_getter('new_train', 'training', 'scaling'),
        apply=_config_path_setter('new_train', 'training', 'scaling'),
    ),
    _OverrideSpec(
        key='loss_name',
        flag='--loss-name',
        positional_index=6,
        coerce=parse_optional_str,
        default_getter=_default_loss_name,
        apply=_config_path_setter('new_train', 'training', 'g_loss_fn'),
    ),
    _OverrideSpec(
        key='dropout_rate',
        flag='--dropout-rate',
        positional_index=7,
        coerce=lambda value: parse_optional_float(value, 'dropout_rate'),
        default_getter=_config_path_getter('new_train', 'network', 'dropout_rate'),
        apply=_config_path_setter('new_train', 'network', 'dropout_rate'),
    ),
    _OverrideSpec(
        key='output_activation',
        flag='--output-activation',
        positional_index=8,
        coerce=parse_optional_str,
        default_getter=_config_path_getter('new_train', 'network', 'output_activation'),
        apply=_config_path_setter('new_train', 'network', 'output_activation'),
    ),
    _OverrideSpec(
        key='kernel_initializer',
        flag='--kernel-initializer',
        positional_index=9,
        coerce=parse_optional_str,
        default_getter=_config_path_getter('new_train', 'network', 'kernel_initializer'),
        apply=_config_path_setter('new_train', 'network', 'kernel_initializer'),
    ),
    _OverrideSpec(
        key='activation_name',
        flag='--activation-name',
        positional_index=10,
        coerce=parse_optional_str,
        default_getter=_config_path_getter('new_train', 'network', 'func'),
        apply=_config_path_setter('new_train', 'network', 'func'),
    ),
    _OverrideSpec(
        key='discriminator_activation',
        flag='--discriminator-activation',
        positional_index=11,
        coerce=parse_optional_str,
        default_getter=_config_path_getter('new_train', 'discriminator', 'func'),
        apply=_config_path_setter('new_train', 'discriminator', 'func'),
    ),
    _OverrideSpec(
        key='discriminator_output_activation',
        flag='--discriminator-output-activation',
        positional_index=12,
        coerce=parse_optional_str,
        default_getter=_config_path_getter('new_train', 'discriminator', 'output_activation'),
        apply=_config_path_setter('new_train', 'discriminator', 'output_activation'),
    ),
    _OverrideSpec(
        key='filter_surveys',
        flag='--filter-surveys',
        positional_index=13,
        coerce=lambda value: parse_optional_bool(value, 'filter_surveys'),
        default_getter=_config_path_getter('create_dataset', 'filter_surveys'),
        apply=_config_path_setter('create_dataset', 'filter_surveys'),
    ),
    _OverrideSpec(
        key='filter_by_last_name',
        flag='--filter-by-last-name',
        positional_index=14,
        coerce=lambda value: parse_optional_bool(value, 'filter_by_last_name'),
        default_getter=_config_path_getter('create_dataset', 'filter_by_last_name'),
        apply=_config_path_setter('create_dataset', 'filter_by_last_name'),
    ),
    _OverrideSpec(
        key='last_name_filter_value',
        flag='--last-name-filter-value',
        positional_index=15,
        coerce=_coerce_last_name_filter_value,
        default_getter=_config_path_getter('create_dataset', 'last_name_filter_value'),
        apply=_config_path_setter('create_dataset', 'last_name_filter_value'),
    ),
)
_OVERRIDE_SPECS_BY_KEY = {spec.key: spec for spec in _OVERRIDE_SPECS}


def _override_keys():
    return tuple(spec.key for spec in _OVERRIDE_SPECS)


def _extract_override_values(source_dict):
    return {spec.key: source_dict.get(spec.key) for spec in _OVERRIDE_SPECS}


@lru_cache(maxsize=1)
def _get_override_arg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    for spec in _OVERRIDE_SPECS:
        parser.add_argument(spec.flag, dest=spec.key)
    return parser


def _read_raw_override_values(argv, start_index):
    cli_tokens = argv[start_index:]
    if any(token.startswith('--') for token in cli_tokens):
        parsed, _ = _get_override_arg_parser().parse_known_args(cli_tokens)
        return _extract_override_values(vars(parsed))

    raw = {}
    for spec in _OVERRIDE_SPECS:
        position = start_index + spec.positional_index
        raw[spec.key] = argv[position] if len(argv) > position and argv[position] else None
    return raw


def _coerce_override_values(raw_overrides):
    return {spec.key: spec.coerce(raw_overrides.get(spec.key)) for spec in _OVERRIDE_SPECS}


def _effective_override_value(config_data, explicit_overrides, key):
    value = explicit_overrides[key]
    if value is not None:
        return value
    return _OVERRIDE_SPECS_BY_KEY[key].default_getter(config_data, explicit_overrides)


# ---------------------------------------------------------------------------
# 2) Compact alias encoding and decoding helpers
# ---------------------------------------------------------------------------

def _pack_u16(value):
    if not isinstance(value, int) or value < 0 or value > 65535:
        raise ValueError(f'Value {value} is out of range for uint16.')
    return value.to_bytes(2, 'big')

def _read_u16(raw, cursor):
    if cursor + 2 > len(raw):
        raise ValueError('Unexpected end of alias token.')
    return int.from_bytes(raw[cursor:cursor + 2], 'big'), cursor + 2

def _pack_string(value):
    encoded = str(value).encode('utf-8')
    return _pack_u16(len(encoded)) + encoded

def _read_string(raw, cursor):
    length, cursor = _read_u16(raw, cursor)
    if cursor + length > len(raw):
        raise ValueError('Unexpected end of alias token while reading string.')
    value = raw[cursor:cursor + length].decode('utf-8')
    return value, cursor + length

def _pack_string_list(values):
    normalized = [str(item) for item in (values or [])]
    if len(normalized) > 255:
        raise ValueError('Too many list items in alias payload.')
    parts = [bytes([len(normalized)])]
    for item in normalized:
        parts.append(_pack_string(item))
    return b''.join(parts)

def _read_string_list(raw, cursor):
    if cursor >= len(raw):
        raise ValueError('Unexpected end of alias token while reading list length.')
    count = raw[cursor]
    cursor += 1
    values = []
    for _ in range(count):
        item, cursor = _read_string(raw, cursor)
        values.append(item)
    return values, cursor

def _pack_tag(value, tag_map):
    normalized = str(value)
    if normalized in tag_map:
        return b'\x01' + bytes([tag_map[normalized]])
    return b'\x00' + _pack_string(normalized)

def _read_tag(raw, cursor, reverse_map):
    if cursor >= len(raw):
        raise ValueError('Unexpected end of alias token while reading tag marker.')
    marker = raw[cursor]
    cursor += 1
    if marker == 1:
        if cursor >= len(raw):
            raise ValueError('Unexpected end of alias token while reading mapped tag.')
        code = raw[cursor]
        cursor += 1
        if code not in reverse_map:
            raise ValueError(f'Unknown mapped tag code: {code}.')
        return reverse_map[code], cursor
    if marker == 0:
        return _read_string(raw, cursor)
    raise ValueError(f'Invalid tag marker: {marker}.')

def _encode_alias_payload(payload_obj):
    """Encode supported payloads into compact reversible URL-safe tokens."""
    if not isinstance(payload_obj, list):
        raise ValueError('Alias payload must be a list.')

    if len(payload_obj) == 7:
        filter_surveys = bool(payload_obj[0])
        allowed_survey = [str(item) for item in (payload_obj[1] or [])]
        filter_by_last_name = bool(payload_obj[2])
        last_name_filter_value = [str(item) for item in (payload_obj[3] or [])] if filter_by_last_name else []
        nsigma = int(payload_obj[4])
        footprint_radius = int(payload_obj[5])
        npixels = int(payload_obj[6])

        flags = 0
        if filter_surveys:
            flags |= 1
        if filter_by_last_name:
            flags |= 2

        parts = [b'D', b'\x01']

        use_mask = filter_surveys and all(item in _KNOWN_SURVEYS for item in allowed_survey)
        if use_mask:
            flags |= 4
        parts.append(bytes([flags]))

        if filter_surveys:
            if use_mask:
                survey_mask = 0
                for survey in allowed_survey:
                    survey_mask |= 1 << _KNOWN_SURVEYS.index(survey)
                parts.append(_pack_u16(survey_mask))
            else:
                parts.append(_pack_string_list(allowed_survey))

        if filter_by_last_name:
            parts.append(_pack_string_list(last_name_filter_value))

        parts.append(_pack_u16(nsigma))
        parts.append(_pack_u16(footprint_radius))
        parts.append(_pack_u16(npixels))

        raw = b''.join(parts)
        compressed = zlib.compress(raw, level=9)
        return base64.urlsafe_b64encode(compressed).decode('ascii').rstrip('=')

    if len(payload_obj) == 10:
        is_gan = str(payload_obj[0])
        use_attention = str(payload_obj[1])
        loss_function = str(payload_obj[2])
        data_alias_enriched_hex = str(payload_obj[3])
        scaling_tag = str(payload_obj[4])
        dropout_tag = str(payload_obj[5])
        activation_tag = str(payload_obj[6])
        output_activation_tag = str(payload_obj[7])
        discriminator_activation_tag = str(payload_obj[8])
        discriminator_output_activation_tag = str(payload_obj[9])

        flags = 0
        if is_gan == 'GAN':
            flags |= 1
        if use_attention == 'ATTN':
            flags |= 2

        if scaling_tag == 'null':
            scaling_tag = 'none'

        parts = [
            b'M',
            b'\x01',
            bytes([flags]),
            _pack_tag(loss_function, _LOSS_CODE_MAP),
            _pack_string(data_alias_enriched_hex),
            _pack_tag(scaling_tag, _SCALING_CODE_MAP),
            _pack_string(dropout_tag),
            _pack_tag(activation_tag, _ACTIVATION_CODE_MAP),
            _pack_tag(output_activation_tag, _ACTIVATION_CODE_MAP),
            _pack_tag(discriminator_activation_tag, _ACTIVATION_CODE_MAP),
            _pack_tag(discriminator_output_activation_tag, _ACTIVATION_CODE_MAP),
        ]
        raw = b''.join(parts)
        compressed = zlib.compress(raw, level=9)
        return base64.urlsafe_b64encode(compressed).decode('ascii').rstrip('=')

    raise ValueError('Unsupported alias payload shape.')

def _decode_alias_payload(token):
    """Decode compact URL-safe alias tokens back to payload lists."""
    if not isinstance(token, str) or token == '':
        raise ValueError(f"Invalid alias token '{token}'.")

    try:
        padded = token + ('=' * (-len(token) % 4))
        compressed = base64.urlsafe_b64decode(padded.encode('ascii'))
        raw = zlib.decompress(compressed)
    except (ValueError, zlib.error) as exc:
        raise ValueError(f"Invalid alias token '{token}'.") from exc

    if len(raw) < 3:
        raise ValueError(f"Invalid alias token '{token}'.")

    kind = chr(raw[0])
    version = raw[1]
    cursor = 2
    if version != 1:
        raise ValueError(f"Invalid alias token '{token}'.")

    if kind == 'D':
        flags = raw[cursor]
        cursor += 1

        filter_surveys = bool(flags & 1)
        filter_by_last_name = bool(flags & 2)
        use_mask = bool(flags & 4)

        allowed_survey = []
        if filter_surveys:
            if use_mask:
                survey_mask, cursor = _read_u16(raw, cursor)
                for index, survey in enumerate(_KNOWN_SURVEYS):
                    if survey_mask & (1 << index):
                        allowed_survey.append(survey)
            else:
                allowed_survey, cursor = _read_string_list(raw, cursor)

        last_name_filter_value = []
        if filter_by_last_name:
            last_name_filter_value, cursor = _read_string_list(raw, cursor)

        nsigma, cursor = _read_u16(raw, cursor)
        footprint_radius, cursor = _read_u16(raw, cursor)
        npixels, cursor = _read_u16(raw, cursor)

        if cursor != len(raw):
            raise ValueError(f"Invalid alias token '{token}'.")

        return [
            1 if filter_surveys else 0,
            allowed_survey,
            1 if filter_by_last_name else 0,
            last_name_filter_value,
            nsigma,
            footprint_radius,
            npixels,
        ]

    if kind == 'M':
        flags = raw[cursor]
        cursor += 1
        is_gan = 'GAN' if (flags & 1) else 'UNET'
        use_attention = 'ATTN' if (flags & 2) else 'NOATTN'

        loss_reverse = {v: k for k, v in _LOSS_CODE_MAP.items()}
        scaling_reverse = {v: k for k, v in _SCALING_CODE_MAP.items()}
        activation_reverse = {v: k for k, v in _ACTIVATION_CODE_MAP.items()}

        loss_function, cursor = _read_tag(raw, cursor, loss_reverse)
        data_alias_enriched_hex, cursor = _read_string(raw, cursor)
        scaling_tag, cursor = _read_tag(raw, cursor, scaling_reverse)
        dropout_tag, cursor = _read_string(raw, cursor)
        activation_tag, cursor = _read_tag(raw, cursor, activation_reverse)
        output_activation_tag, cursor = _read_tag(raw, cursor, activation_reverse)
        discriminator_activation_tag, cursor = _read_tag(raw, cursor, activation_reverse)
        discriminator_output_activation_tag, cursor = _read_tag(raw, cursor, activation_reverse)

        if cursor != len(raw):
            raise ValueError(f"Invalid alias token '{token}'.")

        return [
            is_gan,
            use_attention,
            loss_function,
            data_alias_enriched_hex,
            scaling_tag,
            dropout_tag,
            activation_tag,
            output_activation_tag,
            discriminator_activation_tag,
            discriminator_output_activation_tag,
        ]

    raise ValueError(f"Invalid alias token '{token}'.")

def _component_tag(value):
    """Build a filesystem-safe token for directory/file naming."""
    if value is None:
        return 'none'
    text = str(value).strip()
    if text == '' or text.lower() in {'none', 'null'}:
        return 'none'
    normalized = ''.join(ch.lower() if ch.isalnum() else '_' for ch in text)
    normalized = normalized.strip('_')
    return normalized if normalized else 'none'

def _encode_data_alias_plain(data_alias_plain):
    """Encode data_alias_plain using the compact alias token format."""
    text = '' if data_alias_plain is None else str(data_alias_plain)
    if text == '':
        return ''
    raw = b'A' + b'\x01' + _pack_string(text)
    compressed = zlib.compress(raw, level=9)
    return base64.urlsafe_b64encode(compressed).decode('ascii').rstrip('=')

def _decode_data_alias_plain(data_alias_hex):
    """Decode compact data_alias_hex token back to data_alias_plain."""
    if not isinstance(data_alias_hex, str):
        raise ValueError(f"Invalid data_alias_hex '{data_alias_hex}'.")
    if data_alias_hex == '':
        return ''

    try:
        padded = data_alias_hex + ('=' * (-len(data_alias_hex) % 4))
        compressed = base64.urlsafe_b64decode(padded.encode('ascii'))
        raw = zlib.decompress(compressed)
    except (ValueError, zlib.error) as exc:
        raise ValueError(f"Invalid data_alias_hex '{data_alias_hex}'.") from exc

    if len(raw) < 3 or chr(raw[0]) != 'A' or raw[1] != 1:
        raise ValueError(f"Invalid data_alias_hex '{data_alias_hex}'.")

    value, cursor = _read_string(raw, 2)
    if cursor != len(raw):
        raise ValueError(f"Invalid data_alias_hex '{data_alias_hex}'.")
    return value

def _encode_data_signature(filter_surveys, allowed_survey, filter_by_last_name, last_name_filter_value, nsigma, footprint_radius, npixels):
    """Encode the data signature into a compact reversible token."""
    filter_surveys = bool(filter_surveys)
    filter_by_last_name = bool(filter_by_last_name)
    allowed_survey = [] if allowed_survey is None else ([allowed_survey] if isinstance(allowed_survey, str) else [str(item) for item in allowed_survey])
    last_name_filter_value = [] if last_name_filter_value is None else ([last_name_filter_value] if isinstance(last_name_filter_value, str) else [str(item) for item in last_name_filter_value])
    nsigma = int(nsigma)
    footprint_radius = int(footprint_radius)
    npixels = int(npixels)

    data_alias_plain = ''
    if filter_surveys and allowed_survey:
        data_alias_plain = 'SV_' + '_'.join(allowed_survey)
    if filter_by_last_name and last_name_filter_value:
        data_alias_plain = (data_alias_plain + '_' if data_alias_plain else '') + 'LN' + '_'.join(last_name_filter_value)

    data_alias_hex_code = _encode_data_alias_plain(data_alias_plain)

    data_alias_enriched = f"{data_alias_hex_code}_NS{nsigma}_FP{footprint_radius}_NP{npixels}"
    payload = [
        1 if filter_surveys else 0,
        allowed_survey,
        1 if filter_by_last_name else 0,
        last_name_filter_value,
        nsigma,
        footprint_radius,
        npixels,
    ]
    data_alias_enriched_hex_code = _encode_alias_payload(payload)
    return {
        'data_alias_plain': data_alias_plain,
        'data_alias_hex': data_alias_hex_code,
        'filter_surveys': filter_surveys,
        'allowed_survey': allowed_survey,
        'filter_by_last_name': filter_by_last_name,
        'last_name_filter_value': last_name_filter_value,
        'data_alias_enriched': data_alias_enriched,
        'data_alias_enriched_hex': data_alias_enriched_hex_code,
    }

def _decode_data_signature(data_alias_enriched_hex):
    """Decode the reversible compact token back into all template parameters."""
    if not isinstance(data_alias_enriched_hex, str) or data_alias_enriched_hex == '':
        raise ValueError(f"Invalid data_alias_enriched_hex '{data_alias_enriched_hex}'.")

    payload = _decode_alias_payload(data_alias_enriched_hex)

    if not isinstance(payload, list) or len(payload) != 7:
        raise ValueError('Decoded data_alias_enriched_hex payload must be a 7-item list.')

    filter_surveys = bool(payload[0])
    allowed_survey = [] if payload[1] is None else ([payload[1]] if isinstance(payload[1], str) else [str(item) for item in payload[1]])
    filter_by_last_name = bool(payload[2])
    last_name_filter_value = [] if payload[3] is None else ([payload[3]] if isinstance(payload[3], str) else [str(item) for item in payload[3]])

    try:
        nsigma = int(payload[4])
        footprint_radius = int(payload[5])
        npixels = int(payload[6])
    except (TypeError, ValueError) as exc:
        raise ValueError('Decoded data_alias_enriched_hex payload contains non-integer dataset parameters.') from exc

    data_alias = ''
    if filter_surveys and allowed_survey:
        data_alias = 'SV_' + '_'.join(allowed_survey)
    if filter_by_last_name and last_name_filter_value:
        data_alias = (data_alias + '_' if data_alias else '') + 'LN' + '_'.join(last_name_filter_value)

    data_alias_hex_code = _encode_data_alias_plain(data_alias)
    decoded_data_alias = _decode_data_alias_plain(data_alias_hex_code)
    if decoded_data_alias != data_alias:
        raise ValueError('Decoded data_alias_hex payload is not canonical.')

    data_alias_enriched = f"{data_alias_hex_code}_NS{nsigma}_FP{footprint_radius}_NP{npixels}"
    expected_payload = [
        1 if filter_surveys else 0,
        allowed_survey,
        1 if filter_by_last_name else 0,
        last_name_filter_value,
        nsigma,
        footprint_radius,
        npixels,
    ]
    expected_data_alias_enriched_hex = _encode_alias_payload(expected_payload)
    if expected_data_alias_enriched_hex != data_alias_enriched_hex:
        raise ValueError('Decoded data_alias_enriched_hex payload is not canonical.')

    return {
        'data_alias_hex': data_alias_hex_code,
        'data_alias_plain': data_alias,
        'filter_surveys': filter_surveys,
        'allowed_survey': allowed_survey,
        'filter_by_last_name': filter_by_last_name,
        'last_name_filter_value': last_name_filter_value,
        'nsigma': nsigma,
        'footprint_radius': footprint_radius,
        'npixels': npixels,
        'data_alias_enriched': data_alias_enriched,
        'data_alias_enriched_hex': data_alias_enriched_hex,
    }


# ---------------------------------------------------------------------------
# 3) Template expansion, runtime registry, and path normalization helpers
# ---------------------------------------------------------------------------

def _encode_model_alias(is_gan, use_attention, loss_function, data_alias_enriched_hex,
                         scaling_tag, dropout_tag, activation_tag, output_activation_tag,
                         discriminator_activation_tag, discriminator_output_activation_tag):
    """Encode model configuration into a compact reversible token."""
    model_alias_plain = (
        f"{is_gan}_{use_attention}_{loss_function}_"
        f"{data_alias_enriched_hex}_{scaling_tag}_DO{dropout_tag}_"
        f"ACT{activation_tag}_OUT{output_activation_tag}_"
        f"DACT{discriminator_activation_tag}_DOUT{discriminator_output_activation_tag}"
    )
    payload = [
        is_gan,
        use_attention,
        loss_function,
        data_alias_enriched_hex,
        scaling_tag,
        dropout_tag,
        activation_tag,
        output_activation_tag,
        discriminator_activation_tag,
        discriminator_output_activation_tag,
    ]
    model_alias_hex = _encode_alias_payload(payload)
    return {'model_alias_plain': model_alias_plain, 'model_alias_hex': model_alias_hex}

def _decode_model_alias(model_alias_hex):
    """Decode model_alias_hex into its individual model and data tags."""
    if not isinstance(model_alias_hex, str) or model_alias_hex.strip() == '':
        raise ValueError(f"Invalid model_alias_hex '{model_alias_hex}'.")

    payload = _decode_alias_payload(model_alias_hex)

    if not isinstance(payload, list) or len(payload) != 10:
        raise ValueError('Decoded model_alias_hex payload must be a 10-item list.')

    (is_gan, use_attention, loss_function, data_alias_enriched_hex,
     scaling_tag, dropout_tag, activation_tag, output_activation_tag,
     discriminator_activation_tag, discriminator_output_activation_tag) = payload

    expected_payload = [
        is_gan,
        use_attention,
        loss_function,
        data_alias_enriched_hex,
        scaling_tag,
        dropout_tag,
        activation_tag,
        output_activation_tag,
        discriminator_activation_tag,
        discriminator_output_activation_tag,
    ]
    expected_hex = _encode_alias_payload(expected_payload)
    if expected_hex != model_alias_hex:
        raise ValueError('Decoded model_alias_hex payload is not canonical.')

    model_alias_plain = (
        f"{is_gan}_{use_attention}_{loss_function}_"
        f"{data_alias_enriched_hex}_{scaling_tag}_DO{dropout_tag}_"
        f"ACT{activation_tag}_OUT{output_activation_tag}_"
        f"DACT{discriminator_activation_tag}_DOUT{discriminator_output_activation_tag}"
    )
    dropout_rate = dropout_tag.replace('p', '.')
    try:
        dropout_rate = float(dropout_rate)
    except ValueError:
        pass
    decoded_data = _decode_data_signature(data_alias_enriched_hex)

    return {
        'model_alias_hex': model_alias_hex,
        'model_alias_plain': model_alias_plain,
        'model_type': 'gan' if is_gan == 'GAN' else 'unet',
        'is_gan': is_gan,
        'use_attention': use_attention,
        'attention': use_attention == 'ATTN',
        'loss_function': loss_function,
        'data_alias_enriched_hex': data_alias_enriched_hex,
        'scaling_tag': scaling_tag,
        'dropout_tag': dropout_tag,
        'dropout_rate': dropout_rate,
        'activation_tag': activation_tag,
        'output_activation_tag': output_activation_tag,
        'discriminator_activation_tag': discriminator_activation_tag,
        'discriminator_output_activation_tag': discriminator_output_activation_tag,
        **decoded_data,
    }

def _decode_models_dir(models_dir):
    """Decode models_dir into path/model/data components."""
    if not isinstance(models_dir, str) or models_dir.strip() == '':
        raise ValueError(f"Invalid models_dir '{models_dir}'.")

    normalized = models_dir.strip().replace('\\', '/').rstrip('/')
    match = re.fullmatch(
        r'(?P<models_root_dir>.+)/(?P<is_gan>GAN|UNET)/(?P<use_attention>ATTN|NOATTN)/(?P<loss_function>[^/]+)/(?P<data_alias_enriched_hex>[^/]+)/(?P<scaling_tag>[^/]+)/DO(?P<dropout_tag>[^/]+)/ACT(?P<activation_tag>[^/]+)/OUT(?P<output_activation_tag>[^/]+)/DACT(?P<discriminator_activation_tag>[^/]+)/DOUT(?P<discriminator_output_activation_tag>[^/]+)',
        normalized,
    )
    if match is None:
        raise ValueError(f"Invalid models_dir '{models_dir}'.")

    data_alias_enriched_hex = match.group('data_alias_enriched_hex')
    decoded_data = _decode_data_signature(data_alias_enriched_hex)

    dropout_tag = match.group('dropout_tag')
    dropout_rate = dropout_tag.replace('p', '.')
    try:
        dropout_rate = float(dropout_rate)
    except ValueError:
        pass

    _model_alias_dict = _encode_model_alias(
        match.group('is_gan'), match.group('use_attention'), match.group('loss_function'),
        data_alias_enriched_hex, match.group('scaling_tag'), dropout_tag,
        match.group('activation_tag'), match.group('output_activation_tag'),
        match.group('discriminator_activation_tag'), match.group('discriminator_output_activation_tag'),
    )
    model_alias_hex = _model_alias_dict['model_alias_hex']
    model_alias_plain = _model_alias_dict['model_alias_plain']

    return {
        'models_dir': normalized,
        'models_root_dir': match.group('models_root_dir'),
        'model_alias_hex': model_alias_hex,
        'model_alias_plain': model_alias_plain,
        'model_type': 'gan' if match.group('is_gan') == 'GAN' else 'unet',
        'is_gan': match.group('is_gan'),
        'use_attention': match.group('use_attention'),
        'attention': match.group('use_attention') == 'ATTN',
        'loss_function': match.group('loss_function'),
        'data_alias_enriched_hex': data_alias_enriched_hex,
        'scaling_tag': match.group('scaling_tag'),
        'dropout_tag': dropout_tag,
        'dropout_rate': dropout_rate,
        'activation_tag': match.group('activation_tag'),
        'output_activation_tag': match.group('output_activation_tag'),
        'discriminator_activation_tag': match.group('discriminator_activation_tag'),
        'discriminator_output_activation_tag': match.group('discriminator_output_activation_tag'),
        **decoded_data,
    }

def _format_with_known_templates(text, templates):
    """Format only known replacement fields and preserve unknown placeholders."""
    formatter = string.Formatter()
    formatted_parts = []

    for literal_text, field_name, format_spec, conversion in formatter.parse(text):
        formatted_parts.append(literal_text)
        if field_name is None:
            continue

        token = '{' + field_name
        if conversion:
            token += f'!{conversion}'
        if format_spec:
            token += f':{format_spec}'
        token += '}'

        if field_name not in templates:
            formatted_parts.append(token)
            continue

        value = templates[field_name]
        if conversion == 'r':
            value = repr(value)
        elif conversion == 'a':
            value = ascii(value)
        elif conversion in (None, 's'):
            pass
        else:
            formatted_parts.append(token)
            continue

        try:
            rendered = format(value, format_spec) if format_spec else str(value)
        except (TypeError, ValueError):
            formatted_parts.append(token)
            continue

        formatted_parts.append(rendered)

    return ''.join(formatted_parts)

def _normalize_paths(cfg, templates):
    """Recursively format template strings and absolutize path-like entries.

    Parameters
    ----------
    cfg : Any
        Nested config value (dict/list/scalar).
    templates : dict
        Formatting context used for ``str.format(**templates)``.

    Returns
    -------
    Any
        Deeply normalized config value.
    """
    if isinstance(cfg, dict):
        return {k: _normalize_paths(v, templates) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_normalize_paths(v, templates) for v in cfg]
    if isinstance(cfg, str):
        result = _format_with_known_templates(cfg, templates)
        if result.startswith('./') or result.startswith('../') or result.startswith('/'):
            return os.path.abspath(result)
        return result
    return cfg

def resolve_generator_loss(loss_name, loss_map):
    """Resolve generator loss by name using an explicit loss map."""
    if loss_name not in loss_map:
        raise ValueError(f'Unsupported generator loss: {loss_name}')
    return loss_map[loss_name]

def resolve_gan_loss(loss_name, loss_map):
    """Resolve GAN loss by name using an explicit loss map."""
    if loss_name not in loss_map:
        raise ValueError(f'Unsupported GAN loss: {loss_name}')
    return loss_map[loss_name]

def resolve_data_generator(generator_name, generator_map):
    """Resolve data generator function by name using an explicit generator map."""
    if generator_name not in generator_map:
        raise ValueError(f'Unsupported data generator: {generator_name}')
    return generator_map[generator_name]

@lru_cache(maxsize=1)
def _get_tensorflow_module():
    """Import TensorFlow lazily so simple config operations do not eagerly pay the startup cost."""
    return importlib.import_module('tensorflow')

@lru_cache(maxsize=None)
def _import_runtime_symbol(name):
    """Import one supported project symbol by name."""
    module_name, attr_name = _RUNTIME_IMPORT_SPECS[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)

def _resolve_name_from_runtime_registry(name):
    """Resolve supported config symbols from explicit project and TensorFlow registries."""
    if not isinstance(name, str):
        return name

    if name in _RUNTIME_IMPORT_SPECS:
        return _import_runtime_symbol(name)

    tf_module = _get_tensorflow_module()
    for namespace in (
        tf_module.keras.layers,
        tf_module.keras.optimizers,
        tf_module.keras.losses,
        tf_module.keras.activations,
        tf_module.keras.initializers,
    ):
        namespace_dict = vars(namespace)
        if name in namespace_dict:
            return namespace_dict[name]

    return name

def _resolve_loss_value(loss_value):
    """Resolve config loss value and instantiate keras Loss classes when needed."""
    resolved = _resolve_name_from_runtime_registry(loss_value)
    tf_module = _get_tensorflow_module()

    if isinstance(resolved, type) and issubclass(resolved, tf_module.keras.losses.Loss):
        return resolved()

    return resolved

def _resolve_output_activation_value(value):
    """Resolve output activation while keeping Keras-compatible value types."""
    if value is None:
        return None

    tf_module = _get_tensorflow_module()
    if isinstance(value, str):
        activations_dict = vars(tf_module.keras.activations)
        if value in activations_dict:
            return activations_dict[value]

    return value

def _resolve_initializer_value(value):
    """Resolve initializer and instantiate keras Initializer classes when needed."""
    if value is None:
        return None

    resolved = _resolve_name_from_runtime_registry(value)
    tf_module = _get_tensorflow_module()

    if isinstance(resolved, type) and issubclass(resolved, tf_module.keras.initializers.Initializer):
        return resolved()

    return resolved

@lru_cache(maxsize=1)
def _get_strategy_resolver():
    """Import the training strategy resolver lazily to avoid eager training imports in starter."""
    utils_module = importlib.import_module('src.training.utils')
    return utils_module.resolve_registry_function

def _resolve_data_strategy_value(category, value):
    """Resolve training-data strategy config values into callables in starter."""
    if not isinstance(value, str):
        return value

    if category == 'noise_fn':
        return _resolve_name_from_runtime_registry(value)

    strategy_resolver = _get_strategy_resolver()
    return strategy_resolver(category, value)

def _auto_resolve_training_entries(config_data):
    """Resolve known runtime config entries into objects when possible."""
    if not isinstance(config_data, dict):
        return config_data

    if 'data' in config_data and isinstance(config_data['data'], dict):
        data_cfg = config_data['data']
        sigma_kernel_name = data_cfg['sigma_kernel_fn']
        data_cfg['sigma_kernel_requires_fit_data'] = sigma_kernel_name == 'fit'
        for key in ('candidates_fn', 'post_filter_fn', 'sample_fn', 'sigma_kernel_fn', 'noise_fn', 'stats_name_fn'):
            if key in data_cfg and data_cfg[key] is not None:
                data_cfg[key] = _resolve_data_strategy_value(key, data_cfg[key])

    if 'network' in config_data and isinstance(config_data['network'], dict) and 'func' in config_data['network']:
        network_cfg = config_data['network']
        network_cfg['func'] = _resolve_name_from_runtime_registry(network_cfg['func'])

    if 'network' in config_data and isinstance(config_data['network'], dict):
        network_cfg = config_data['network']
        if 'output_activation' in network_cfg and network_cfg['output_activation'] is not None:
            network_cfg['output_activation'] = _resolve_output_activation_value(network_cfg['output_activation'])
        if 'kernel_initializer' in network_cfg and network_cfg['kernel_initializer'] is not None:
            network_cfg['kernel_initializer'] = _resolve_initializer_value(network_cfg['kernel_initializer'])

    if 'discriminator' in config_data and isinstance(config_data['discriminator'], dict) and 'func' in config_data['discriminator']:
        discriminator_cfg = config_data['discriminator']
        discriminator_cfg['func'] = _resolve_name_from_runtime_registry(discriminator_cfg['func'])

    if 'training' in config_data and isinstance(config_data['training'], dict):
        training_cfg = config_data['training']
        if 'optimizer' in training_cfg:
            training_cfg['optimizer'] = _resolve_name_from_runtime_registry(training_cfg['optimizer'])
        if 'g_loss_fn' in training_cfg:
            training_cfg['g_loss_fn'] = _resolve_loss_value(training_cfg['g_loss_fn'])
        if 'data_generator' in training_cfg:
            training_cfg['data_generator'] = _resolve_name_from_runtime_registry(training_cfg['data_generator'])

    if 'gan' in config_data and isinstance(config_data['gan'], dict) and 'loss_fn' in config_data['gan']:
        gan_cfg = config_data['gan']
        gan_cfg['loss_fn'] = _resolve_loss_value(gan_cfg['loss_fn'])

    return config_data


# ---------------------------------------------------------------------------
# 4) Override application and template-context construction
# ---------------------------------------------------------------------------

def parse_config_overrides(args=None, argv=None, start_index=1):
    """Parse optional naming/config overrides for :func:`load_config`.

    Returns a plain dict with optional values (None means "use config.yaml").
    """
    if args is not None:
        raw = _extract_override_values(vars(args))
    else:
        argv = sys.argv if argv is None else argv
        raw = _read_raw_override_values(argv, start_index)
    return _coerce_override_values(raw)

def _override_config_with_explicit_values(config_data, explicit_overrides):
    for spec in _OVERRIDE_SPECS:
        value = explicit_overrides[spec.key]
        if value is not None:
            spec.apply(config_data, value)
    return config_data

def _build_template_context(config_data, explicit_overrides):
    """Build a small formatting context used by path/file templates."""
    dataset_cfg = config_data['create_dataset']
    new_train_cfg = config_data['new_train']
    train_cfg = new_train_cfg['training']
    network_cfg = new_train_cfg['network']
    gan_cfg = new_train_cfg['gan']
    time_tag = datetime.now().strftime('%Y-%m-%d-%H-%M')

    filter_surveys = _effective_override_value(config_data, explicit_overrides, 'filter_surveys')
    filter_by_last_name = _effective_override_value(config_data, explicit_overrides, 'filter_by_last_name')
    last_name_filter_value = _effective_override_value(config_data, explicit_overrides, 'last_name_filter_value')

    nsigma = _effective_override_value(config_data, explicit_overrides, 'nsigma')
    footprint_radius = _effective_override_value(config_data, explicit_overrides, 'footprint_radius')
    npixels = _effective_override_value(config_data, explicit_overrides, 'npixels')
    
    data_dict = _encode_data_signature(
        filter_surveys=filter_surveys,
        allowed_survey=dataset_cfg['allowed_survey'],
        filter_by_last_name=filter_by_last_name,
        last_name_filter_value=last_name_filter_value,
        nsigma=nsigma,
        footprint_radius=footprint_radius,
        npixels=npixels,
    )
    data_alias_plain = data_dict['data_alias_plain']
    data_alias_hex = data_dict['data_alias_hex']
    data_alias_enriched_plain = data_dict['data_alias_enriched']
    data_alias_enriched_hex = data_dict['data_alias_enriched_hex']

    model_type = _effective_override_value(config_data, explicit_overrides, 'model_type')
    attention = bool(_effective_override_value(config_data, explicit_overrides, 'attention'))
    is_gan = 'GAN' if str(model_type).lower() == 'gan' else 'UNET'
    use_attention = 'ATTN' if attention else 'NOATTN'
    scaling = _effective_override_value(config_data, explicit_overrides, 'scaling')
    loss_name = _effective_override_value(config_data, explicit_overrides, 'loss_name')
    dropout_rate = _effective_override_value(config_data, explicit_overrides, 'dropout_rate')
    activation_name = _effective_override_value(config_data, explicit_overrides, 'activation_name')
    output_activation = _effective_override_value(config_data, explicit_overrides, 'output_activation')
    discriminator_activation = _effective_override_value(config_data, explicit_overrides, 'discriminator_activation')
    discriminator_output_activation = _effective_override_value(config_data, explicit_overrides, 'discriminator_output_activation')
    _loss_aliases = {
        'MeanSquaredError':      'MSE',
        'MeanAbsoluteError':     'MAE',
        'ssim_loss':             'SSIM',
        'scale_invariant_mae':   'SIMAE',
        'log_cosh_loss':         'LOGCOSH',
        'BinaryCrossentropy':    'BCE',
    }
    loss_function = _loss_aliases[loss_name] if loss_name in _loss_aliases else str(loss_name)
    scaling_tag = str(scaling)
    dropout_tag = str(dropout_rate).replace('.', 'p')
    activation_tag = _component_tag(activation_name)
    output_activation_tag = _component_tag(output_activation)
    discriminator_activation_tag = _component_tag(discriminator_activation)
    discriminator_output_activation_tag = _component_tag(discriminator_output_activation)
    
    data_dir = f"{config_data['paths']['data_dir']}/{data_alias_enriched_plain}"
    _model_alias_dict = _encode_model_alias(
        is_gan, use_attention, loss_function, data_alias_enriched_hex,
        scaling_tag, dropout_tag, activation_tag, output_activation_tag,
        discriminator_activation_tag, discriminator_output_activation_tag,
    )
    model_alias_hex = _model_alias_dict['model_alias_hex']
    model_alias_plain = _model_alias_dict['model_alias_plain']
    models_dir = f"""{config_data['paths']['models_dir']}/
            {is_gan}/{use_attention}/{loss_function}/
            {data_alias_enriched_hex}/{scaling_tag}/DO{dropout_tag}/
            ACT{activation_tag}/OUT{output_activation_tag}/
            DACT{discriminator_activation_tag}/DOUT{discriminator_output_activation_tag}
            """.replace('\n', '').replace(' ', '')

    return {
        'data_alias_enriched_hex': data_alias_enriched_hex,
        'nsigma': nsigma,
        'footprint_radius': footprint_radius,
        'npixels': npixels,
        'data_dir': data_dir,
        'models_dir': models_dir,
        'multimodal_metrics_dir': config_data['paths']['multimodal_metrics_dir'],
        'singlemodal_metrics_dir': config_data['paths']['singlemodal_metrics_dir'],
        'plots_dir': config_data['paths']['plots_dir'],
        'model_type': model_type,
        'is_gan': is_gan,
        'use_attention': use_attention,
        'scaling': scaling,
        'scaling_tag': scaling_tag,
        'loss_name': loss_name,
        'loss_function': loss_function,
        'dropout_rate': dropout_rate,
        'dropout_tag': dropout_tag,
        'activation_name': activation_name,
        'output_activation': output_activation,
        'discriminator_activation': discriminator_activation,
        'discriminator_output_activation': discriminator_output_activation,
        'activation_tag': activation_tag,
        'output_activation_tag': output_activation_tag,
        'discriminator_activation_tag': discriminator_activation_tag,
        'discriminator_output_activation_tag': discriminator_output_activation_tag,
        'model_alias_plain': model_alias_plain,
        'model_alias_hex': model_alias_hex,
        'time_tag': time_tag,
    }


def _normalize_config_tree(config_data, template_context, label):
    """Normalize templates and synchronize computed path roots for one config tree."""
    normalized = _require_mapping(_normalize_paths(config_data, template_context), label)
    paths_cfg = _require_mapping(normalized['paths'], f'{label}["paths"]')
    for key in ('data_dir', 'models_dir', 'multimodal_metrics_dir', 'singlemodal_metrics_dir', 'plots_dir'):
        paths_cfg[key] = _normalize_paths(template_context[key], template_context)
    return normalized


_MISSING = object()
_RUNTIME_SOURCE_PATTERN = re.compile(r"resolved from '([^']+)' at runtime", re.IGNORECASE)
_RESOLUTION_ORDER = (
    'mast',
    'create_dataset',
    'new_train',
    'metrics',
    'uncropped_metrics',
    'merge_catalogs',
    'prepare_images',
    'prepare_plots',
)


# ---------------------------------------------------------------------------
# 5) Runtime-source extraction and section-level value resolution
# ---------------------------------------------------------------------------


def _require_mapping(value, path):
    if not isinstance(value, dict):
        raise TypeError(f'{path} must be a mapping.')
    return value


def _path_get(mapping, path, *, default=_MISSING):
    current = mapping
    for part in path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _write_path_value(mapping, path, value):
    parts = path.split('.')
    current = mapping
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            raise ConfigResolutionError(f'Cannot assign {path!r}: {part!r} is not a mapping.')
        current = current[part]

    current[parts[-1]] = copy.deepcopy(value)


def _section_has_forced_values(section_name, forced_paths):
    prefix = f'{section_name}.'
    return any(path.startswith(prefix) for path in forced_paths)


def _parse_runtime_source_comment(comment):
    match = _RUNTIME_SOURCE_PATTERN.search(comment or '')
    if match is None:
        return ()
    refs_text = match.group(1).replace(' and ', ',')
    return tuple(part.strip() for part in refs_text.split(',') if part.strip())


def _extract_runtime_sources(config_text):
    runtime_sources = {}
    stack = []

    for line in config_text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#') or stripped.startswith('-'):
            continue

        indent = len(line) - len(stripped)
        while stack and stack[-1][0] >= indent:
            stack.pop()

        if ':' not in stripped:
            continue

        key, rest = stripped.split(':', 1)
        key = key.strip()
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            continue

        value_part, comment = rest, ''
        if '#' in rest:
            value_part, comment = rest.split('#', 1)
            comment = comment.strip()
        value = value_part.strip()

        path = '.'.join([part for _, part in stack] + [key])
        if value == '':
            stack.append((indent, key))
            continue

        refs = _parse_runtime_source_comment(comment)
        if value == 'null' and refs:
            runtime_sources[path] = refs

    return runtime_sources


@lru_cache(maxsize=8)
def _read_runtime_sources_from_path(path_str):
    path = Path(path_str)
    if not path.exists():
        return {}
    return _extract_runtime_sources(path.read_text(encoding='utf-8'))


def _get_runtime_sources(config_path):
    runtime_sources = dict(_read_runtime_sources_from_path(str(CONFIG_PATH)))
    runtime_sources.update(_read_runtime_sources_from_path(str(config_path)))
    return runtime_sources


def _collect_changed_runtime_refs(baseline_config, overridden_config, runtime_sources):
    changed_refs = set()
    for source_paths in runtime_sources.values():
        for source_path in source_paths:
            if _path_get(baseline_config, source_path, default=_MISSING) != _path_get(overridden_config, source_path, default=_MISSING):
                changed_refs.add(source_path)
    return changed_refs


def _apply_runtime_transform(target_path, source_values):
    leaf = target_path.rsplit('.', 1)[-1]

    if target_path == 'create_dataset.split_dirs':
        return list(source_values)

    if leaf in {'patch_size', 'uncropped_patch_size'}:
        value = source_values[0]
        if not isinstance(value, int):
            raise ConfigResolutionError(f'{target_path!r} requires an integer source value.')
        return [value, value, 1]

    if leaf == 'model_prototype':
        value = source_values[0]
        if not isinstance(value, str) or not value:
            raise ConfigResolutionError(f'{target_path!r} requires a non-empty string source value.')
        return re.sub(r'\{epoch[^}]*\}', '*', value).replace('{prefix}', '*')

    if len(source_values) == 1:
        return source_values[0]
    return list(source_values)


def _resolve_section_runtime_values(section_name, section_cfg, resolved_root, runtime_sources, forced_paths):
    prefix = f'{section_name}.'
    current_root = dict(resolved_root)
    current_root[section_name] = section_cfg

    for full_target_path, source_paths in runtime_sources.items():
        if not full_target_path.startswith(prefix):
            continue

        target_path = full_target_path[len(prefix):]
        current_value = _path_get(section_cfg, target_path, default=_MISSING)
        if current_value is _MISSING:
            raise ConfigResolutionError(f'Missing required config key: {full_target_path!r}.')

        source_is_forced = any(source_path in forced_paths for source_path in source_paths)
        if current_value is not None and not source_is_forced:
            continue

        source_values = []
        for source_path in source_paths:
            source_value = _path_get(current_root, source_path, default=_MISSING)
            if source_value is _MISSING:
                raise ConfigResolutionError(f'Unknown runtime source {source_path!r} for {full_target_path!r}.')
            source_values.append(copy.deepcopy(source_value))

        _write_path_value(section_cfg, target_path, _apply_runtime_transform(full_target_path, source_values))
        if source_is_forced:
            forced_paths.add(full_target_path)


def _validate_and_sync_checkpoint_config(config_data):
    """Validate checkpoint settings for the resolved new_train section."""
    training_cfg = _require_mapping(config_data.get('training'), 'config["training"]')
    data_cfg = _require_mapping(config_data.get('data'), 'config["data"]')

    checkpoint_filename_pattern = training_cfg['checkpoint_filename_pattern']
    if not isinstance(checkpoint_filename_pattern, str) or not checkpoint_filename_pattern:
        raise ValueError('training.checkpoint_filename_pattern must be a non-empty string.')

    restore_kwargs = training_cfg['checkpoint_restore_kwargs']
    if not isinstance(restore_kwargs, dict):
        raise TypeError('training.checkpoint_restore_kwargs must be a mapping.')
    restore_kwargs = dict(restore_kwargs)

    if 'filename_pattern' not in restore_kwargs:
        raise ValueError("training.checkpoint_restore_kwargs must include 'filename_pattern'.")
    if restore_kwargs['filename_pattern'] != checkpoint_filename_pattern:
        raise ValueError('training.checkpoint_restore_kwargs.filename_pattern must match training.checkpoint_filename_pattern.')

    training_cfg['checkpoint_restore_kwargs'] = restore_kwargs
    data_cfg['checkpoint_restore_kwargs'] = copy.deepcopy(restore_kwargs)
    data_cfg['checkpoint_custom_epoch'] = training_cfg['checkpoint_custom_epoch']
    return config_data


def _finalize_new_train_section(section_cfg, _resolved_root):
    _validate_and_sync_checkpoint_config(section_cfg)
    _auto_resolve_training_entries(section_cfg)

    training_cfg = section_cfg['training']
    training_cfg['data_kwargs'] = copy.deepcopy(section_cfg['data'])
    training_cfg['network_kwargs'] = copy.deepcopy(section_cfg['network'])
    training_cfg['discriminator_kwargs'] = copy.deepcopy(section_cfg['discriminator'])
    training_cfg['gan_kwargs'] = copy.deepcopy(section_cfg['gan'])


def _finalize_metrics_section(section_cfg, resolved_root):
    data_cfg = copy.deepcopy(resolved_root['new_train']['data'])
    create_dataset_cfg = resolved_root['create_dataset']

    section_cfg['func'] = _resolve_name_from_runtime_registry(section_cfg['func'])
    noise_fn = data_cfg['noise_fn']

    section_cfg['data_kwargs'] = {
        'kwargs_data': data_cfg,
    }
    if section_cfg['use_custom_test_images']:
        section_cfg['data_kwargs']['kwargs_data']['metadata_filepath'] = create_dataset_cfg['noisy_filtered_metadata_output_file']

    section_cfg['model_kwargs'] = {
        'patch_size': tuple(section_cfg['patch_size']),
        'stride': tuple(section_cfg['stride']),
        'weighting': section_cfg['weighting'],
        'batch_size': section_cfg['batch_size'],
        'gaussian_sigma': section_cfg['gaussian_sigma'],
        'type_of_image': section_cfg['type_of_image'],
        'nan_value': data_cfg['nan_value'],
        'posinf_value': data_cfg['posinf_value'],
        'neginf_value': data_cfg['neginf_value'],
        'location_col': data_cfg['location_col'],
        'exp_time_col': data_cfg['exposure_col'],
        'new_exp_time_col': section_cfg['new_exp_time_col'],
        'sigma_key': data_cfg['sigma_key'],
        'noise_fn': noise_fn,
        'combined_images_dir': section_cfg['combined_images_dir'],
        'png_dir': section_cfg['png_dir'],
        'org_dir': section_cfg['org_dir'],
        'noisy_dir': section_cfg['noisy_dir'],
        'rec_dir': section_cfg['rec_dir'],
        'use_mosaic': section_cfg['use_mosaic'],
    }

    source_keys = [
        'sigma', 'maxiters', 'nsigma', 'npixels', 'nlevels', 'contrast',
        'footprint_radius', 'distance_threshold', 'deblend', 'deblend_timeout',
        'alpha', 'beta', 'gamma', 'k1', 'k2', 'win_size', 'win_sigma',
        'func', 'thresh', 'org_thresh', 'radius_factor', 'PHOT_FLUXFRAC',
        'r_min', 'elongation_fraction', 'PHOT_AUTOPARAMS', 'maskthresh',
        'minarea', 'org_minarea', 'filter_type', 'deblend_nthresh', 'deblend_cont',
        'clean', 'clean_param',
    ]
    section_cfg['kwargs_source'] = {key: copy.deepcopy(section_cfg[key]) for key in source_keys if key in section_cfg}
    section_cfg['kwargs_source']['sigma_key'] = data_cfg['sigma_key']
    section_cfg['kwargs_source']['noise_fn'] = noise_fn
    section_cfg['kwargs_source']['type_of_image'] = section_cfg['type_of_image']
    section_cfg['kwargs_source']['nan_value'] = data_cfg['nan_value']
    section_cfg['kwargs_source']['posinf_value'] = data_cfg['posinf_value']
    section_cfg['kwargs_source']['neginf_value'] = data_cfg['neginf_value']


def _finalize_uncropped_metrics_section(section_cfg, _resolved_root):
    kwargs_source = copy.deepcopy(section_cfg['kwargs_source'])
    kwargs_source['uncropped_patch_size'] = copy.deepcopy(section_cfg['uncropped_patch_size'])
    kwargs_source['uncropped_stride'] = copy.deepcopy(section_cfg['uncropped_stride'])
    kwargs_source['uncropped_weighting'] = section_cfg['uncropped_weighting']
    kwargs_source['uncropped_batch_size'] = section_cfg['uncropped_batch_size']
    kwargs_source['uncropped_use_mosaic'] = section_cfg['uncropped_use_mosaic']
    section_cfg['kwargs_source'] = kwargs_source
    section_cfg['data_kwargs'] = copy.deepcopy(section_cfg['data_kwargs'])


# ---------------------------------------------------------------------------
# 6) Section finalizers and public config loading API
# ---------------------------------------------------------------------------


_SECTION_FINALIZERS = {
    'new_train': _finalize_new_train_section,
    'metrics': _finalize_metrics_section,
    'uncropped_metrics': _finalize_uncropped_metrics_section,
}

_DERIVED_FORCED_OUTPUTS = {
    'new_train': (
        'training.data_kwargs',
        'training.network_kwargs',
        'training.discriminator_kwargs',
        'training.gan_kwargs',
    ),
    'metrics': (
        'data_kwargs',
        'model_kwargs',
        'kwargs_source',
    ),
    'uncropped_metrics': (
        'kwargs_source',
        'data_kwargs',
    ),
}


def resolve_config(config_data, *, runtime_sources: dict[str, tuple[str, ...]] | None = None, forced_paths: set[str] | None = None):
    """Resolve null placeholders into a per-script runtime config root."""
    if not isinstance(config_data, dict):
        raise TypeError('config.yaml must contain a top-level mapping.')

    if 'paths' not in config_data:
        raise ConfigResolutionError("Missing required top-level config key: 'paths'")

    runtime_sources = dict(_get_runtime_sources(CONFIG_PATH) if runtime_sources is None else runtime_sources)
    resolved_config = {
        'paths': copy.deepcopy(_require_mapping(config_data['paths'], 'config["paths"]')),
    }
    forced_paths = set[str]() if forced_paths is None else set(forced_paths)

    for section_name in _RESOLUTION_ORDER:
        if section_name not in config_data:
            raise ConfigResolutionError(f'Missing required top-level config key: {section_name!r}')
        section_cfg = copy.deepcopy(_require_mapping(config_data[section_name], f'config[{section_name!r}]'))
        _resolve_section_runtime_values(section_name, section_cfg, resolved_config, runtime_sources, forced_paths)

        current_root = dict(resolved_config)
        current_root[section_name] = section_cfg
        finalizer = _SECTION_FINALIZERS.get(section_name)
        if finalizer is not None:
            finalizer(section_cfg, current_root)
            if _section_has_forced_values(section_name, forced_paths):
                for output_path in _DERIVED_FORCED_OUTPUTS.get(section_name, ()): 
                    forced_paths.add(f'{section_name}.{output_path}')

        resolved_config[section_name] = section_cfg

    return resolved_config


def load_config(
    config_path=None,
    *,
    nsigma=None,
    footprint_radius=None,
    npixels=None,
    model_type=None,
    attention=None,
    scaling=None,
    loss_name=None,
    dropout_rate=None,
    output_activation=None,
    kernel_initializer=None,
    activation_name=None,
    discriminator_activation=None,
    discriminator_output_activation=None,
    filter_surveys=None,
    filter_by_last_name=None,
    last_name_filter_value=None,
) -> dict[str, Any]:
    """Load config, apply overrides, normalize templates, and resolve script sections."""
    # Keep explicit mapping so override parameters are visibly consumed.
    override_inputs = {
        'nsigma': nsigma,
        'footprint_radius': footprint_radius,
        'npixels': npixels,
        'model_type': model_type,
        'attention': attention,
        'scaling': scaling,
        'loss_name': loss_name,
        'dropout_rate': dropout_rate,
        'output_activation': output_activation,
        'kernel_initializer': kernel_initializer,
        'activation_name': activation_name,
        'discriminator_activation': discriminator_activation,
        'discriminator_output_activation': discriminator_output_activation,
        'filter_surveys': filter_surveys,
        'filter_by_last_name': filter_by_last_name,
        'last_name_filter_value': last_name_filter_value,
    }
    path = Path(config_path) if config_path is not None else CONFIG_PATH

    config_text = path.read_text(encoding='utf-8')
    config_data = yaml.safe_load(config_text)

    if not isinstance(config_data, dict):
        raise TypeError('config.yaml must contain a top-level mapping.')

    runtime_sources = _get_runtime_sources(path)
    cli_overrides = _extract_override_values(override_inputs)

    baseline_config = copy.deepcopy(config_data)
    empty_overrides = {key: None for key in cli_overrides}
    baseline_template_context = _build_template_context(baseline_config, empty_overrides)
    baseline_config = _normalize_config_tree(baseline_config, baseline_template_context, 'normalized baseline config')

    config_data = _override_config_with_explicit_values(config_data, cli_overrides)
    template_context = _build_template_context(config_data, cli_overrides)
    config_data = _normalize_config_tree(config_data, template_context, 'normalized config')
    forced_paths = _collect_changed_runtime_refs(baseline_config, config_data, runtime_sources)
    return resolve_config(config_data, runtime_sources=runtime_sources, forced_paths=forced_paths)


