"""Bounded, read-only snapshot status client; no response text enters errors."""
import json
import math
from http.client import HTTPException
from urllib.request import Request, build_opener

from .client_url import client_base_url
from .client_response import validate_content_encoding, validate_json_content_type, validate_transfer_encoding
from .credentials import environment_key
from .embedding import NoRedirect
from .errors import ConfigError
from .jsonutil import loads

MAX_RESPONSE = 64 * 1024


class SnapshotStatusError(Exception):
    """Remote request or response failure, safe for CLI reporting."""


def snapshot_status(endpoint, knowledge_id, *, key_env='BRIDGE_API_KEY', timeout=10):
    """Read the named environment key and return only a validated active status."""
    url = client_base_url(endpoint, context='Snapshot status') + '/snapshot/status'
    if not isinstance(knowledge_id, str) or not 1 <= len(knowledge_id) <= 256:
        raise ConfigError('Snapshot status knowledge_id must contain 1..256 characters')
    key = environment_key(key_env, context='Snapshot status')
    try:
        timeout = float(timeout)
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ConfigError('Snapshot status timeout must be finite and in (0,120]') from None

    req = Request(url, data=json.dumps({'knowledge_id': knowledge_id}).encode(),
                  headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key},
                  method='POST')
    try:
        with build_opener(NoRedirect()).open(req, timeout=timeout) as response:
            validate_transfer_encoding(response.headers)
            validate_json_content_type(response.headers)
            validate_content_encoding(response.headers)
            if response.status != 200:
                raise ValueError
            raw = response.read(MAX_RESPONSE + 1)
            lengths = response.headers.get_all('Content-Length', [])
            if (len(raw) > MAX_RESPONSE or len(lengths) > 1 or
                    (lengths and (not lengths[0].isascii() or not lengths[0].isdecimal() or
                                  int(lengths[0]) != len(raw)))):
                raise ValueError
            body = loads(raw.decode('utf-8'))
        if (not isinstance(body, dict) or set(body) != {'knowledge_id', 'version', 'record_count'} or
                type(body['knowledge_id']) is not str or body['knowledge_id'] != knowledge_id or
                type(body['version']) is not int or body['version'] < 1 or
                type(body['record_count']) is not int or body['record_count'] < 0):
            raise ValueError
        return body
    except (OSError, ValueError, HTTPException, RecursionError):
        raise SnapshotStatusError('Snapshot status failed: request rejected, unavailable or invalid response') from None
