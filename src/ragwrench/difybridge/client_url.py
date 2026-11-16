"""Shared strict base-URL preflight for read-only CLI clients."""
import ipaddress
import re
from urllib.parse import urlsplit

from .config import http_url
from .errors import ConfigError


def client_base_url(endpoint, *, context):
    try:
        if (not isinstance(endpoint, str) or not endpoint.isascii() or
                any(ord(c) <= 32 or ord(c) >= 127 for c in endpoint) or
                any(c in endpoint for c in ('?', '#', '\\'))):
            raise ValueError
        base = http_url(endpoint)
        url = urlsplit(base)
        if '@' in url.netloc or (url.port is not None and not 1 <= url.port <= 65535):
            raise ValueError
        host = url.hostname
        if url.netloc.startswith('['):
            ipaddress.IPv6Address(host)
        elif not re.fullmatch(r'[A-Za-z0-9.-]+', host):
            raise ValueError
        return base
    except (ValueError, ConfigError):
        raise ConfigError(f'{context} requires a valid ASCII HTTP(S) base URL without credentials/query/fragment') from None
