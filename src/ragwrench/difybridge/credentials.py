"""Shared client credential preflight; errors never include names or values."""
import os
import re

from .errors import ConfigError


def validate_key(key, *, context):
    if (not isinstance(key, str) or not 1 <= len(key) <= 4096 or
            not key.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in key)):
        raise ConfigError(f'{context} key must be set to a visible ASCII token without whitespace (max 4096 characters)')
    return key


def environment_key(name, *, context):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        raise ConfigError(f'{context} requires a valid environment variable name')
    return validate_key(os.environ.get(name), context=context)
