"""Minimal config loader for config.yaml."""

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
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml

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

def _parse_optional_int(value, field_name):
    """Parse optional integer overrides."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {field_name} '{value}'. Must be an integer.")

def _parse_optional_float(value, field_name):
    """Parse optional float overrides."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {field_name} '{value}'. Must be numeric.")

def _parse_optional_bool(value, field_name):
    """Parse optional boolean overrides from common truthy/falsey strings."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f"Invalid {field_name} '{value}'. Must be a boolean value.")

def _parse_optional_str(value):
    """Parse optional string overrides and normalize common null-like values."""
    if value is None:
        return None
    parsed = str(value).strip()
    if parsed == '' or parsed.lower() in {'none', 'null'}:
        return None
    return parsed

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
        sigma_kernel_name = data_cfg.get('sigma_kernel_fn')
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

def parse_config_overrides(args=None, argv=None, start_index=1):
    """Parse optional naming/config overrides for :func:`load_config`.

    Returns a plain dict with optional values (None means "use config.yaml").
    """
    keys = (
        'nsigma', 'footprint_radius', 'npixels',
        'model_type', 'attention', 'scaling', 'loss_name', 'dropout_rate',
        'output_activation', 'kernel_initializer',
        'activation_name', 'discriminator_activation', 'discriminator_output_activation',
        'filter_surveys', 'filter_by_last_name', 'last_name_filter_value',
    )

    if args is not None:
        args_dict = vars(args)
        raw = {
            key: (args_dict[key] if key in args_dict else None) for key in keys
        }
    else:
        argv = sys.argv if argv is None else argv
        cli_tokens = argv[start_index:]

        # Named flags are preferred; keep positional parsing for backward compatibility.
        if any(token.startswith('--') for token in cli_tokens):
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('--nsigma', dest='nsigma')
            parser.add_argument('--footprint-radius', dest='footprint_radius')
            parser.add_argument('--npixels', dest='npixels')
            parser.add_argument('--model-type', dest='model_type')
            
            parser.add_argument('--attention', dest='attention')
            parser.add_argument('--scaling', dest='scaling')
            parser.add_argument('--loss-name', dest='loss_name')
            parser.add_argument('--dropout-rate', dest='dropout_rate')
            parser.add_argument('--output-activation', dest='output_activation')
            parser.add_argument('--kernel-initializer', dest='kernel_initializer')
            parser.add_argument('--activation-name', dest='activation_name')
            parser.add_argument('--discriminator-activation', dest='discriminator_activation')
            parser.add_argument('--discriminator-output-activation', dest='discriminator_output_activation')
            parser.add_argument('--filter-surveys', dest='filter_surveys')
            parser.add_argument('--filter-by-last-name', dest='filter_by_last_name')
            parser.add_argument('--last-name-filter-value', dest='last_name_filter_value')
            parsed, _ = parser.parse_known_args(cli_tokens)
            parsed_dict = vars(parsed)
            raw = {key: (parsed_dict[key] if key in parsed_dict else None) for key in keys}
        else:
            raw = {
                'nsigma': argv[start_index] if len(argv) > start_index and argv[start_index] else None,
                'footprint_radius': argv[start_index + 1] if len(argv) > start_index + 1 and argv[start_index + 1] else None,
                'npixels': argv[start_index + 2] if len(argv) > start_index + 2 and argv[start_index + 2] else None,
                'model_type': argv[start_index + 3] if len(argv) > start_index + 3 and argv[start_index + 3] else None,
                'attention': argv[start_index + 4] if len(argv) > start_index + 4 and argv[start_index + 4] else None,
                'scaling': argv[start_index + 5] if len(argv) > start_index + 5 and argv[start_index + 5] else None,
                'loss_name': argv[start_index + 6] if len(argv) > start_index + 6 and argv[start_index + 6] else None,
                'dropout_rate': argv[start_index + 7] if len(argv) > start_index + 7 and argv[start_index + 7] else None,
                'output_activation': argv[start_index + 8] if len(argv) > start_index + 8 and argv[start_index + 8] else None,
                'kernel_initializer': argv[start_index + 9] if len(argv) > start_index + 9 and argv[start_index + 9] else None,
                'activation_name': argv[start_index + 10] if len(argv) > start_index + 10 and argv[start_index + 10] else None,
                'discriminator_activation': argv[start_index + 11] if len(argv) > start_index + 11 and argv[start_index + 11] else None,
                'discriminator_output_activation': argv[start_index + 12] if len(argv) > start_index + 12 and argv[start_index + 12] else None,
                'filter_surveys': argv[start_index + 13] if len(argv) > start_index + 13 and argv[start_index + 13] else None,
                'filter_by_last_name': argv[start_index + 14] if len(argv) > start_index + 14 and argv[start_index + 14] else None,
                'last_name_filter_value': argv[start_index + 15] if len(argv) > start_index + 15 and argv[start_index + 15] else None,
            }
    last_names = _parse_optional_str(raw['last_name_filter_value'])
    if last_names is not None:
        last_names = [name.strip() for name in last_names.split(',') if name.strip()]
    return {
        'nsigma': _parse_optional_int(raw['nsigma'], 'nsigma'),
        'footprint_radius': _parse_optional_int(raw['footprint_radius'], 'footprint_radius'),
        'npixels': _parse_optional_int(raw['npixels'], 'npixels'),
        'model_type': _parse_optional_str(raw['model_type']),
        'attention': _parse_optional_bool(raw['attention'], 'attention'),
        'scaling': _parse_optional_str(raw['scaling']),
        'loss_name': _parse_optional_str(raw['loss_name']),
        'dropout_rate': _parse_optional_float(raw['dropout_rate'], 'dropout_rate'),
        'output_activation': _parse_optional_str(raw['output_activation']),
        
        'kernel_initializer': _parse_optional_str(raw['kernel_initializer']),
        'activation_name': _parse_optional_str(raw['activation_name']),
        'discriminator_activation': _parse_optional_str(raw['discriminator_activation']),
        'discriminator_output_activation': _parse_optional_str(raw['discriminator_output_activation']),
        
        'filter_surveys': _parse_optional_bool(raw['filter_surveys'], 'filter_surveys'),
        'filter_by_last_name': _parse_optional_bool(raw['filter_by_last_name'], 'filter_by_last_name'),
        'last_name_filter_value': last_names,

    }

def _override_config_with_explicit_values(config_data, explicit_overrides): 
    config_data['create_dataset']['nsigma'] = explicit_overrides['nsigma'] if explicit_overrides['nsigma'] is not None else config_data['create_dataset']['nsigma']
    config_data['create_dataset']['footprint_radius'] = explicit_overrides['footprint_radius'] if explicit_overrides['footprint_radius'] is not None else config_data['create_dataset']['footprint_radius']
    config_data['create_dataset']['npixels'] = explicit_overrides['npixels'] if explicit_overrides['npixels'] is not None else config_data['create_dataset']['npixels']
    config_data['training']['use_gan'] = explicit_overrides['model_type'].lower() == 'gan' if explicit_overrides['model_type'] is not None else config_data['training']['use_gan']
    config_data['network']['attention'] = explicit_overrides['attention'] if explicit_overrides['attention'] is not None else config_data['network']['attention']
    config_data['training']['scaling'] = explicit_overrides['scaling'] if explicit_overrides['scaling'] is not None else config_data['training']['scaling']
    config_data['training']['g_loss_fn'] = explicit_overrides['loss_name'] if explicit_overrides['loss_name'] is not None else config_data['training']['g_loss_fn']
    config_data['network']['dropout_rate'] = explicit_overrides['dropout_rate'] if explicit_overrides['dropout_rate'] is not None else config_data['network']['dropout_rate']
    config_data['network']['output_activation'] = explicit_overrides['output_activation'] if explicit_overrides['output_activation'] is not None else config_data['network']['output_activation']

    config_data['network']['kernel_initializer'] = explicit_overrides['kernel_initializer'] if explicit_overrides['kernel_initializer'] is not None else config_data['network']['kernel_initializer']
    config_data['network']['func'] = explicit_overrides['activation_name'] if explicit_overrides['activation_name'] is not None else config_data['network']['func']
    config_data['discriminator']['func'] = explicit_overrides['discriminator_activation'] if explicit_overrides['discriminator_activation'] is not None else config_data['discriminator']['func']
    config_data['discriminator']['output_activation'] = explicit_overrides['discriminator_output_activation'] if explicit_overrides['discriminator_output_activation'] is not None else config_data['discriminator']['output_activation']

    config_data['create_dataset']['filter_surveys'] = explicit_overrides['filter_surveys'] if explicit_overrides['filter_surveys'] is not None else config_data['create_dataset']['filter_surveys']
    config_data['create_dataset']['filter_by_last_name'] = explicit_overrides['filter_by_last_name'] if explicit_overrides['filter_by_last_name'] is not None else config_data['create_dataset']['filter_by_last_name']
    config_data['create_dataset']['last_name_filter_value'] = explicit_overrides['last_name_filter_value'] if explicit_overrides['last_name_filter_value'] is not None else config_data['create_dataset']['last_name_filter_value']
    return config_data

def _build_template_context(config_data, explicit_overrides):
    """Build a small formatting context used by path/file templates."""
    dataset_cfg = config_data['create_dataset']
    train_cfg = config_data['training']
    network_cfg = config_data['network']
    gan_cfg = config_data['gan']
    time_tag = datetime.now().strftime('%Y-%m-%d-%H-%M')

    filter_surveys = explicit_overrides['filter_surveys'] if explicit_overrides['filter_surveys'] is not None else dataset_cfg['filter_surveys']
    filter_by_last_name = explicit_overrides['filter_by_last_name'] if explicit_overrides['filter_by_last_name'] is not None else dataset_cfg['filter_by_last_name']
    last_name_filter_value = explicit_overrides['last_name_filter_value'] if explicit_overrides['last_name_filter_value'] is not None else dataset_cfg['last_name_filter_value']
    
    nsigma = explicit_overrides['nsigma'] if explicit_overrides['nsigma'] is not None else dataset_cfg['nsigma']
    footprint_radius = explicit_overrides['footprint_radius'] if explicit_overrides['footprint_radius'] is not None else dataset_cfg['footprint_radius']
    npixels = explicit_overrides['npixels'] if explicit_overrides['npixels'] is not None else dataset_cfg['npixels']
    
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

    model_type = explicit_overrides['model_type'] if explicit_overrides['model_type'] is not None else ('gan' if train_cfg['use_gan'] else 'unet')
    attention = explicit_overrides['attention'] if explicit_overrides['attention'] is not None else bool(network_cfg['attention'])
    is_gan = 'GAN' if str(model_type).lower() == 'gan' else 'UNET'
    use_attention = 'ATTN' if attention else 'NOATTN'
    scaling = explicit_overrides['scaling'] if explicit_overrides['scaling'] is not None else train_cfg['scaling']
    loss_name = explicit_overrides['loss_name'] if explicit_overrides['loss_name'] is not None else (gan_cfg['loss_fn'] if str(model_type).lower() == 'gan' else train_cfg['g_loss_fn'])
    dropout_rate = explicit_overrides['dropout_rate'] if explicit_overrides['dropout_rate'] is not None else network_cfg['dropout_rate']
    activation_name = explicit_overrides['activation_name'] if explicit_overrides['activation_name'] is not None else network_cfg['func']
    output_activation = explicit_overrides['output_activation'] if explicit_overrides['output_activation'] is not None else network_cfg['output_activation']
    discriminator_activation = explicit_overrides['discriminator_activation'] if explicit_overrides['discriminator_activation'] is not None else config_data['discriminator']['func']
    discriminator_output_activation = explicit_overrides['discriminator_output_activation'] if explicit_overrides['discriminator_output_activation'] is not None else config_data['discriminator']['output_activation']
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

def _build_evaluation_kwargs(config_data):
    """Pre-assemble evaluation kwargs dicts and store them inside *config_data['evaluation']*.

    Parameters
    ----------
    config_data : dict
        Fully loaded and path-normalised config dictionary.

    Notes
    -----
    Three sub-dicts are written into ``config_data['evaluation']``:

    * ``data_kwargs`` — forwarded to :func:`~src.evaluation.metrics.get_test_images`.
    * ``model_kwargs`` — forwarded through
      :func:`~src.evaluation.metrics.process_models` down to
      :func:`~src.evaluation.metrics.process_single_model`.
    * ``kwargs_source`` — source-detection, photometry and SSIM parameters
      forwarded to :func:`~src.evaluation.metrics.compare_images` and
      :func:`~src.evaluation.uncropped_metrics.process_subdf`.  Includes the
      ``uncropped_*`` sliding-window keys required by
      :mod:`~src.evaluation.uncropped_metrics`.
    """
    eval_cfg     = config_data['evaluation']
    data_cfg     = config_data['data']
    training_cfg = config_data['training']

    # Resolve func and noise_fn from string names to callables when possible.
    eval_cfg['func'] = _resolve_name_from_runtime_registry(eval_cfg['func'])
    noise_fn = _resolve_name_from_runtime_registry(data_cfg['noise_fn'])

    # data_kwargs ────────────────────────────────────────────────────────────
    eval_cfg['data_kwargs'] = {
        'kwargs_data': data_cfg.copy(), # copy to avoid mutating original config entries
    }
    if eval_cfg['use_custom_test_images']:
        eval_cfg['data_kwargs']['kwargs_data']['metadata_filepath'] = config_data['create_dataset']['noisy_filtered_metadata_output_file']

    # model_kwargs ───────────────────────────────────────────────────────────
    eval_cfg['model_kwargs'] = {
        'patch_size':          tuple(eval_cfg['patch_size']),
        'stride':              tuple(eval_cfg['stride']),
        'weighting':           eval_cfg['weighting'],
        'batch_size':          eval_cfg['batch_size'],
        'gaussian_sigma':      eval_cfg['gaussian_sigma'],
        'type_of_image':       eval_cfg['type_of_image'],
        'nan_value':           data_cfg['nan_value'],
        'posinf_value':        data_cfg['posinf_value'],
        'neginf_value':        data_cfg['neginf_value'],
        'location_col':        data_cfg['location_col'],
        'exp_time_col':        data_cfg['exposure_col'],
        'new_exp_time_col':    eval_cfg['new_exp_time_col'],
        'sigma_key':           data_cfg['sigma_key'],
        'noise_fn':            noise_fn,
        'combined_images_dir': eval_cfg['combined_images_dir'],
        'png_dir':             eval_cfg['png_dir'],
        'org_dir':             eval_cfg['org_dir'],
        'noisy_dir':           eval_cfg['noisy_dir'],
        'rec_dir':             eval_cfg['rec_dir'],
        'use_mosaic':          eval_cfg['use_mosaic'],
    }

    # kwargs_source ──────────────────────────────────────────────────────────
    _source_keys = [
        'sigma', 'maxiters', 'nsigma', 'npixels', 'nlevels', 'contrast',
        'footprint_radius', 'distance_threshold', 'deblend', 'deblend_timeout',
        'alpha', 'beta', 'gamma', 'k1', 'k2', 'win_size', 'win_sigma',
        'func', 'thresh', 'org_thresh', 'radius_factor', 'PHOT_FLUXFRAC',
        'r_min', 'elongation_fraction', 'PHOT_AUTOPARAMS', 'maskthresh',
        'minarea', 'org_minarea', 'filter_type', 'deblend_nthresh', 'deblend_cont',
        'clean', 'clean_param',
        # uncropped (Faber) sliding-window inference parameters
        'uncropped_patch_size', 'uncropped_stride', 'uncropped_weighting', 'uncropped_batch_size',
    ]
    eval_cfg['kwargs_source'] = {k: eval_cfg[k] for k in _source_keys if k in eval_cfg}
    eval_cfg['kwargs_source']['sigma_key'] = data_cfg['sigma_key']
    eval_cfg['kwargs_source']['noise_fn'] = noise_fn
    eval_cfg['kwargs_source']['type_of_image'] = eval_cfg['type_of_image']
    eval_cfg['kwargs_source']['nan_value'] = data_cfg['nan_value']
    eval_cfg['kwargs_source']['posinf_value'] = data_cfg['posinf_value']
    eval_cfg['kwargs_source']['neginf_value'] = data_cfg['neginf_value']

    eval_cfg['filter_by_last_name'] = config_data['create_dataset']['filter_by_last_name']
    eval_cfg['last_name_filter_value'] = config_data['create_dataset']['last_name_filter_value']
    eval_cfg['last_name_col'] = config_data['create_dataset']['last_name_col']


def _get_shared_binding_policy(config_data):
    """Return shared binding policy (defaults to preserving explicit config values)."""
    shared_bindings_cfg = config_data.get('shared_bindings', {})
    if not isinstance(shared_bindings_cfg, dict):
        return {'prefer_explicit_values': True}
    return {
        'prefer_explicit_values': bool(shared_bindings_cfg.get('prefer_explicit_values', True)),
    }

def _ensure_runtime_shared_groups(config_data):
    """Create grouped runtime-shared metadata that records derived vs explicit values."""
    runtime_cfg = config_data.setdefault('runtime', {})
    shared_cfg = runtime_cfg.setdefault('shared', {})
    for group_name in (
        'checkpoint',
        'paths',
        'data_dataset_mast',
        'training_evaluation',
        'visualization',
    ):
        shared_cfg.setdefault(group_name, {})
    return shared_cfg

def _bind_shared_value(config_data, section, key, derived_value, *, group_name, policy, source_path):
    """Bind one shared value with optional explicit override and track its origin."""
    section_cfg = config_data[section]
    prefer_explicit_values = policy['prefer_explicit_values']
    has_explicit = key in section_cfg and section_cfg[key] is not None

    if prefer_explicit_values and has_explicit:
        value = section_cfg[key]
        origin = 'explicit'
    else:
        section_cfg[key] = derived_value
        value = derived_value
        origin = 'derived'

    runtime_shared = _ensure_runtime_shared_groups(config_data)
    runtime_shared[group_name][f'{section}.{key}'] = {
        'value': value,
        'origin': origin,
        'source': source_path,
    }
    return value

def _bind_nested_shared_value(config_data, section, subsection, key, derived_value, *, group_name, policy, source_path):
    """Bind one shared value for section/subsection/key with explicit-override support."""
    section_cfg = config_data[section]
    subsection_cfg = section_cfg.setdefault(subsection, {})
    if not isinstance(subsection_cfg, dict):
        raise TypeError(f'config[{section!r}][{subsection!r}] must be a mapping for shared bindings.')

    prefer_explicit_values = policy['prefer_explicit_values']
    has_explicit = key in subsection_cfg and subsection_cfg[key] is not None

    if prefer_explicit_values and has_explicit:
        value = subsection_cfg[key]
        origin = 'explicit'
    else:
        subsection_cfg[key] = derived_value
        value = derived_value
        origin = 'derived'

    runtime_shared = _ensure_runtime_shared_groups(config_data)
    runtime_shared[group_name][f'{section}.{subsection}.{key}'] = {
        'value': value,
        'origin': origin,
        'source': source_path,
    }
    return value

def _validate_and_sync_checkpoint_config(config_data):
    """Validate checkpoint settings once and share them across consumers."""
    policy = _get_shared_binding_policy(config_data)
    training_cfg = config_data['training']

    checkpoint_filename_pattern = training_cfg['checkpoint_filename_pattern']
    if not isinstance(checkpoint_filename_pattern, str) or not checkpoint_filename_pattern:
        raise ValueError("training.checkpoint_filename_pattern must be a non-empty string.")

    restore_kwargs = training_cfg['checkpoint_restore_kwargs']
    if isinstance(restore_kwargs, dict):
        restore_kwargs = dict(restore_kwargs)
    else:
        raise TypeError("training.checkpoint_restore_kwargs must be a mapping.")

    if 'filename_pattern' not in restore_kwargs:
        raise ValueError("training.checkpoint_restore_kwargs must include 'filename_pattern'.")
    if restore_kwargs['filename_pattern'] != checkpoint_filename_pattern:
        raise ValueError("training.checkpoint_restore_kwargs.filename_pattern must match training.checkpoint_filename_pattern.")

    training_cfg['checkpoint_restore_kwargs'] = restore_kwargs
    _bind_shared_value(
        config_data,
        'data',
        'checkpoint_restore_kwargs',
        restore_kwargs,
        group_name='checkpoint',
        policy=policy,
        source_path='training.checkpoint_restore_kwargs',
    )
    _bind_shared_value(
        config_data,
        'data',
        'checkpoint_custom_epoch',
        training_cfg['checkpoint_custom_epoch'],
        group_name='checkpoint',
        policy=policy,
        source_path='training.checkpoint_custom_epoch',
    )

    model_prototype = re.sub(r'\{epoch[^}]*\}', '*', checkpoint_filename_pattern).replace('{prefix}', '*')
    _bind_shared_value(
        config_data,
        'evaluation',
        'model_prototype',
        model_prototype,
        group_name='checkpoint',
        policy=policy,
        source_path='training.checkpoint_filename_pattern',
    )

def _sync_path_bindings(config_data):
    """Bind canonical paths consumed by data, mast, training and visualization scripts."""
    policy = _get_shared_binding_policy(config_data)
    paths_cfg = config_data['paths']

    _bind_shared_value(config_data, 'data', 'training_path', paths_cfg['training_path'], group_name='paths', policy=policy, source_path='paths.training_path')
    _bind_shared_value(config_data, 'data', 'eval_path', paths_cfg['eval_path'], group_name='paths', policy=policy, source_path='paths.eval_path')
    _bind_shared_value(config_data, 'data', 'fit_data_filepath', paths_cfg['fit_info_csv'], group_name='paths', policy=policy, source_path='paths.fit_info_csv')

    _bind_shared_value(config_data, 'mast', 'metadata_output', paths_cfg['metadata_csv'], group_name='paths', policy=policy, source_path='paths.metadata_csv')
    _bind_shared_value(config_data, 'create_dataset', 'metadata_filepath', paths_cfg['metadata_csv'], group_name='paths', policy=policy, source_path='paths.metadata_csv')
    _bind_shared_value(config_data, 'create_dataset', 'dataset_dir', paths_cfg['data_dir'], group_name='paths', policy=policy, source_path='paths.data_dir')
    split_dirs = [
        paths_cfg['training_path'],
        paths_cfg['eval_path'],
        paths_cfg['test_path'],
    ]
    _bind_shared_value(config_data, 'create_dataset', 'split_dirs', split_dirs, group_name='paths', policy=policy, source_path='paths.training_path|paths.eval_path|paths.test_path')

    _bind_shared_value(config_data, 'training', 'training_results_dir', paths_cfg['models_dir'], group_name='paths', policy=policy, source_path='paths.models_dir')
    _bind_shared_value(config_data, 'evaluation', 'models_dir', paths_cfg['models_dir'], group_name='paths', policy=policy, source_path='paths.models_dir')

    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'model_dir', paths_cfg['models_dir'], group_name='paths', policy=policy, source_path='paths.models_dir')

def _sync_data_dataset_mast_contract(config_data):
    """Share schema and preprocessing parameters used jointly by data ingestion scripts."""
    policy = _get_shared_binding_policy(config_data)
    data_cfg = config_data['data']

    _bind_shared_value(config_data, 'data', 'metadata_filepath', config_data['create_dataset']['cropped_stats_output_file'], group_name='data_dataset_mast', policy=policy, source_path='create_dataset.cropped_stats_output_file')
    _bind_shared_value(config_data, 'evaluation', 'metadata_filepath', config_data['create_dataset']['cropped_stats_output_file'], group_name='data_dataset_mast', policy=policy, source_path='create_dataset.cropped_stats_output_file')
    _bind_shared_value(config_data, 'data', 'stats_column_map', config_data['create_dataset']['stats_column_map'], group_name='data_dataset_mast', policy=policy, source_path='create_dataset.stats_column_map')
    _bind_shared_value(config_data, 'data', 'original_stats_prefix', config_data['create_dataset']['original_stats_prefix'], group_name='data_dataset_mast', policy=policy, source_path='create_dataset.original_stats_prefix')

    _bind_shared_value(config_data, 'create_dataset', 'id_column', data_cfg['dataset'], group_name='data_dataset_mast', policy=policy, source_path='data.dataset')
    _bind_shared_value(config_data, 'mast', 'id_column', data_cfg['dataset'], group_name='data_dataset_mast', policy=policy, source_path='data.dataset')
    _bind_shared_value(config_data, 'create_dataset', 'url_column', config_data['mast']['url_column'], group_name='data_dataset_mast', policy=policy, source_path='mast.url_column')
    _bind_shared_value(config_data, 'create_dataset', 'exp_column', data_cfg['exposure_col'], group_name='data_dataset_mast', policy=policy, source_path='data.exposure_col')
    _bind_shared_value(config_data, 'mast', 'main_column', data_cfg['exposure_col'], group_name='data_dataset_mast', policy=policy, source_path='data.exposure_col')
    _bind_shared_value(config_data, 'create_dataset', 'type_of_image', data_cfg['type_of_image'], group_name='data_dataset_mast', policy=policy, source_path='data.type_of_image')
    _bind_shared_value(config_data, 'evaluation', 'type_of_image', data_cfg['type_of_image'], group_name='data_dataset_mast', policy=policy, source_path='data.type_of_image')

    _bind_shared_value(config_data, 'create_dataset', 'nan_value', data_cfg['nan_value'], group_name='data_dataset_mast', policy=policy, source_path='data.nan_value')
    _bind_shared_value(config_data, 'create_dataset', 'posinf_value', data_cfg['posinf_value'], group_name='data_dataset_mast', policy=policy, source_path='data.posinf_value')
    _bind_shared_value(config_data, 'create_dataset', 'neginf_value', data_cfg['neginf_value'], group_name='data_dataset_mast', policy=policy, source_path='data.neginf_value')
    _bind_shared_value(config_data, 'create_dataset', 'location_col', data_cfg['location_col'], group_name='data_dataset_mast', policy=policy, source_path='data.location_col')
    _bind_shared_value(config_data, 'create_dataset', 'ps', data_cfg['ps'], group_name='data_dataset_mast', policy=policy, source_path='data.ps')

    _bind_shared_value(config_data, 'create_dataset', 'reset_after', config_data['mast']['reset_after'], group_name='data_dataset_mast', policy=policy, source_path='mast.reset_after')
    _bind_shared_value(config_data, 'create_dataset', 'max_requests', config_data['mast']['max_requests'], group_name='data_dataset_mast', policy=policy, source_path='mast.max_requests')


def _sync_training_evaluation_contract(config_data):
    """Share patching and detection hyperparameters between training and evaluation."""
    policy = _get_shared_binding_policy(config_data)
    data_cfg = config_data['data']

    patch_size = [data_cfg['ps'], data_cfg['ps'], 1]
    training_patch_size = _bind_shared_value(config_data, 'training', 'patch_size', patch_size, group_name='training_evaluation', policy=policy, source_path='data.ps')

    _bind_shared_value(config_data, 'evaluation', 'patch_size', training_patch_size, group_name='training_evaluation', policy=policy, source_path='training.patch_size')
    _bind_shared_value(config_data, 'evaluation', 'batch_size', config_data['training']['batch_size'], group_name='training_evaluation', policy=policy, source_path='training.batch_size')
    _bind_shared_value(config_data, 'evaluation', 'scaling', config_data['training']['scaling'], group_name='training_evaluation', policy=policy, source_path='training.scaling')
    for key in ('sigma', 'nsigma', 'npixels', 'footprint_radius', 'maxiters'):
        _bind_shared_value(config_data, 'evaluation', key, config_data['create_dataset'][key], group_name='training_evaluation', policy=policy, source_path=f'create_dataset.{key}')
    _bind_shared_value(config_data, 'evaluation', 'hist_min_exp', data_cfg['lowest_power'], group_name='training_evaluation', policy=policy, source_path='data.lowest_power')
    _bind_shared_value(config_data, 'evaluation', 'hist_max_exp', data_cfg['highest_power'], group_name='training_evaluation', policy=policy, source_path='data.highest_power')
    _bind_shared_value(config_data, 'evaluation', 'uncropped_patch_size', config_data['training']['patch_size'], group_name='training_evaluation', policy=policy, source_path='training.patch_size')
    _bind_shared_value(config_data, 'evaluation', 'uncropped_stride', config_data['evaluation']['stride'], group_name='training_evaluation', policy=policy, source_path='evaluation.stride')
    _bind_shared_value(config_data, 'evaluation', 'uncropped_weighting', config_data['evaluation']['weighting'], group_name='training_evaluation', policy=policy, source_path='evaluation.weighting')

def _sync_visualization_contract(config_data):
    """Propagate runtime visualization defaults from data settings."""
    policy = _get_shared_binding_policy(config_data)
    data_cfg = config_data['data']
    paths_cfg = config_data['paths']
    eval_cfg = config_data['evaluation']
    training_cfg = config_data['training']

    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'low', data_cfg['low'], group_name='visualization', policy=policy, source_path='data.low')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'dataset', data_cfg['dataset'], group_name='visualization', policy=policy, source_path='data.dataset')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'metadata_filepath', paths_cfg['metadata_csv'], group_name='visualization', policy=policy, source_path='paths.metadata_csv')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'ps', data_cfg['ps'], group_name='visualization', policy=policy, source_path='data.ps')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'type_of_image', data_cfg['type_of_image'], group_name='visualization', policy=policy, source_path='data.type_of_image')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'exp_column', data_cfg['exposure_col'], group_name='visualization', policy=policy, source_path='data.exposure_col')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'model_prototype', eval_cfg['model_prototype'], group_name='visualization', policy=policy, source_path='evaluation.model_prototype')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'scaling', training_cfg['scaling'], group_name='visualization', policy=policy, source_path='training.scaling')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'ratio_initial', data_cfg['ratio_initial'], group_name='visualization', policy=policy, source_path='data.ratio_initial')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'ratio_count', data_cfg['ratio_count'], group_name='visualization', policy=policy, source_path='data.ratio_count')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_images', 'ratio_growth', data_cfg['ratio_growth'], group_name='visualization', policy=policy, source_path='data.ratio_growth')

    _bind_nested_shared_value(config_data, 'visualization', 'prepare_plots', 'metadata_filepath', config_data['create_dataset']['cropped_stats_output_file'], group_name='visualization', policy=policy, source_path='create_dataset.cropped_stats_output_file')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_plots', 'uncropped_output_dir', eval_cfg['uncropped_output_dir'], group_name='visualization', policy=policy, source_path='evaluation.uncropped_output_dir')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_plots', 'photometrical_data_filename', eval_cfg['photometrical_data_filename'], group_name='visualization', policy=policy, source_path='evaluation.photometrical_data_filename')
    _bind_nested_shared_value(config_data, 'visualization', 'prepare_plots', 'uncropped_results_csv', eval_cfg['uncropped_results_csv'], group_name='visualization', policy=policy, source_path='evaluation.uncropped_results_csv')

def _apply_shared_runtime_bindings(config_data):
    """Apply grouped cross-section runtime bindings used by multiple scripts."""
    policy = _get_shared_binding_policy(config_data)
    config_data.setdefault('runtime', {})['shared_binding_policy'] = dict(policy)

    _validate_and_sync_checkpoint_config(config_data)
    _sync_path_bindings(config_data)
    _sync_data_dataset_mast_contract(config_data)
    _sync_training_evaluation_contract(config_data)
    _sync_visualization_contract(config_data)

def _sync_post_eval_runtime_bindings(config_data):
    """Bind values that depend on evaluation kwargs built later in load_config."""
    policy = _get_shared_binding_policy(config_data)
    _bind_nested_shared_value(
        config_data,
        'visualization',
        'prepare_images',
        'kwargs_source',
        config_data['evaluation']['kwargs_source'],
        group_name='visualization',
        policy=policy,
        source_path='evaluation.kwargs_source',
    )
    _bind_nested_shared_value(
        config_data,
        'visualization',
        'prepare_images',
        'data_kwargs',
        dict(config_data['data']),
        group_name='visualization',
        policy=policy,
        source_path='data.*',
    )

def _sync_post_training_runtime_bindings(config_data):
    """Bind training-local payloads that originate from other config sections."""
    policy = _get_shared_binding_policy(config_data)
    _bind_shared_value(
        config_data,
        'training',
        'data_kwargs',
        dict(config_data['data']),
        group_name='training_evaluation',
        policy=policy,
        source_path='data.*',
    )
    _bind_shared_value(
        config_data,
        'training',
        'network_kwargs',
        dict(config_data['network']),
        group_name='training_evaluation',
        policy=policy,
        source_path='network.*',
    )
    _bind_shared_value(
        config_data,
        'training',
        'discriminator_kwargs',
        dict(config_data['discriminator']),
        group_name='training_evaluation',
        policy=policy,
        source_path='discriminator.*',
    )
    _bind_shared_value(
        config_data,
        'training',
        'gan_kwargs',
        dict(config_data['gan']),
        group_name='training_evaluation',
        policy=policy,
        source_path='gan.*',
    )


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
    """Load config, optionally merge overrides, and normalize path-like fields."""
    path = Path(config_path) if config_path is not None else CONFIG_PATH

    with path.open('r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    if not isinstance(config_data, dict):
        raise TypeError('config.yaml must contain a top-level mapping.')

    cli_overrides = {
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

    config_data = _override_config_with_explicit_values(config_data, cli_overrides)
    template_context = _build_template_context(config_data, cli_overrides)
    config_data = _normalize_paths(config_data, template_context)
    config_data['paths']['models_dir'] = template_context['models_dir']
    config_data['paths']['data_dir'] = template_context['data_dir']
    config_data['paths']['multimodal_metrics_dir'] = template_context['multimodal_metrics_dir']
    config_data['paths']['singlemodal_metrics_dir'] = template_context['multimodal_metrics_dir']
    config_data['paths']['plots_dir'] = template_context['plots_dir'] 
    #_dump_data_alias_map(config_data['paths']['models_dir'])
    _apply_shared_runtime_bindings(config_data)
    
    config_data = _auto_resolve_training_entries(config_data)
    _sync_post_training_runtime_bindings(config_data)
    _build_evaluation_kwargs(config_data)
    _sync_post_eval_runtime_bindings(config_data)
    return config_data


