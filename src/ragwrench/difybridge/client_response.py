"""Strict response headers for read-only checker and snapshot clients."""

import re


_TOKEN = r"[!#$%&'*+.^_`|~0-9A-Za-z-]+"
_QUOTED = r'"(?:[\t\x20-\x21\x23-\x5b\x5d-\x7e]|\\[\t\x20-\x7e])*"'
_JSON_TYPE = re.compile(
    r'[ \t]*application/json[ \t]*'
    rf'(?:;[ \t]*{_TOKEN}[ \t]*=[ \t]*(?:{_TOKEN}|{_QUOTED})[ \t]*)*',
    re.IGNORECASE | re.ASCII,
)


def validate_json_content_type(headers):
    """Require one JSON type; allow token/quoted parameters, never a type list."""
    values = headers.get_all('Content-Type', [])
    if len(values) != 1 or _JSON_TYPE.fullmatch(values[0]) is None:
        raise ValueError('Invalid response Content-Type')


def validate_content_encoding(headers):
    """No decompression: allow absence or one bare case-insensitive identity."""
    values = headers.get_all('Content-Encoding', [])
    if values and (len(values) != 1 or values[0].lower() != 'identity'):
        raise ValueError('Invalid response Content-Encoding')


def validate_transfer_encoding(headers):
    """Allow only one bare chunked declaration, without any Content-Length."""
    transfers = headers.get_all('Transfer-Encoding', [])
    if transfers and (len(transfers) != 1 or transfers[0].lower() != 'chunked' or
                      headers.get_all('Content-Length', [])):
        raise ValueError('Invalid response transfer framing')
