"""Shared parsing helpers for CLI overrides and strict config coercion."""

from __future__ import annotations


def parse_optional_int(value, field_name):
    """Parse optional integer overrides."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid {field_name} '{value}'. Must be an integer.") from err


def parse_optional_float(value, field_name):
    """Parse optional float overrides."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid {field_name} '{value}'. Must be numeric.") from err


def parse_optional_bool(value, field_name):
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


def parse_optional_str(value):
    """Parse optional string overrides and normalize common null-like values."""
    if value is None:
        return None
    parsed = str(value).strip()
    if parsed == '' or parsed.lower() in {'none', 'null'}:
        return None
    return parsed


def parse_required_int(value, field_name):
    """Parse required integer-like config values (accepts ints or numeric strings)."""
    if value is None:
        raise ValueError(f"Missing required {field_name}.")
    try:
        return int(float(value))
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid {field_name} '{value}'. Must be integer-like.") from err
