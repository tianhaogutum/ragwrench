"""Strict JSON input shared by request, corpus, embedding and checker readers."""
import json


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON constant")


def loads(raw):
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
