"""Unambiguous JSON decoding shared by interchange and provenance readers."""
import pytest

from ragwrench.highlight.core import parse_json


@pytest.mark.parametrize("payload, expected", [
    ('{"items":[{"action":"accept"},{"action":"skip"}]}',
     {"items": [{"action": "accept"}, {"action": "skip"}]}),
    ('{"message":"\\\"annotations\\\": [], \\\"annotations\\\": null","annotations":[]}',
     {"message": '"annotations": [], "annotations": null', "annotations": []}),
    (b'{"annotations":[],"Annotations":null}', {"annotations": [], "Annotations": None}),
])
def test_unique_keys_are_per_object_and_ignore_string_contents(payload, expected):
    assert parse_json(payload) == expected
