import json
from pathlib import Path

import pytest

from ragwrench.difybridge.config import ConfigError, http_url, load_config
from ragwrench.difybridge.service import Application


@pytest.mark.parametrize('text', [
    '', '[knowledge]', '[knowledge.x]\nbackend="nope"',
    '[knowledge.x]\nbackend="jsonl"\npath="missing"',
    '[knowledge.x]\nbackend="qdrant"\ncollection="c"',
    '[knowledge.x]\nbackend="jsonl"\npath="data.jsonl"\nmetadata_allowlist=["a.b"]',
    '[knowledge.x]\nbackend="jsonl"\npath="data.jsonl"\nmetadata_allowlist=["a","a"]',
    '[knowledge.x]\nbackend="jsonl"\npath="data.jsonl"\ncontent_field="a.b"',
    '[knowledge.x]\nbackend="jsonl"\npath="data.jsonl"\noverfetch=0',
    '[knowledge.x]\nbackend="jsonl"\npath="data.jsonl"\nmax_candidates=true',
    '[knowledge.x]\nbackend="jsonl"\npath="data.jsonl"\nscore_scale=nan',
    '[knowledge.x]\nbackend="jsonl"\npath="data.jsonl"\ntimeout=-1',
    '[knowledge.x]\nbackend="jsonl"\npath="data.jsonl"\nunknown=true',
    '[knowledge.x]\nbackend="jsonl"\npath="data.jsonl"\ncollection="c"',
    '[knowledge.x]\nbackend="qdrant"\nurl="http://localhost:6333"\ncollection="c"\ndistance="manhattan"',
    '[knowledge.x]\nbackend="qdrant"\nurl="http://localhost:6333"\ncollection="c"\nembedder={kind="script",path="evil.py"}',
    '[knowledge.x]\nbackend="qdrant"\nurl="http://localhost:6333"\ncollection="c"\nembedder={kind="hash",dimensions=0}',
    '[knowledge.x]\nbackend="qdrant"\nurl="http://localhost:6333"\ncollection="c"\nembedder={kind="http",url="http://user:secret@localhost/embeddings",model="x"}',
    '[knowledge.x]\nbackend="qdrant"\nurl="http://localhost:6333"\ncollection="c"\nembedder={kind="http",url="http://localhost/embeddings",model="x",timeout=0}',
])
def test_invalid_config(tmp_path, text):
    (tmp_path / 'data.jsonl').write_text('{"content":"hello"}\n')
    config = tmp_path / 'bridge.toml'
    config.write_text(text)
    with pytest.raises(ConfigError):
        load_config(config)


@pytest.mark.parametrize('line', ['null', '{}', '{"content":"x","metadata":null}', '{"content":"x","metadata":[]}', '{broken}', '{"content":"x","content":"y"}', '{"content":"x","title":2}', '\n'])
def test_invalid_jsonl_startup(tmp_path, line):
    (tmp_path / 'chunks.jsonl').write_text(line + '\n')
    config = tmp_path / 'bridge.toml'
    config.write_text('[knowledge.x]\nbackend="jsonl"\npath="chunks.jsonl"')
    with pytest.raises(ConfigError):
        Application(load_config(config), 'key')


def test_relative_paths_and_field_mapping(tmp_path, monkeypatch):
    (tmp_path / 'chunks.jsonl').write_text(json.dumps({'text': 'mapped vacation policy', 'name': 'source', 'attrs': {'public': 'visible', 'private': 'hide'}}) + '\n')
    config = tmp_path / 'bridge.toml'
    config.write_text('''[knowledge.x]
backend="jsonl"
path="chunks.jsonl"
content_field="text"
title_field="name"
metadata_field="attrs"
metadata_allowlist=["public"]
''')
    monkeypatch.chdir(tmp_path.parent)
    app = Application(load_config(config), 'key')
    try:
        result = app.retrieve('Bearer key', b'{"knowledge_id":"x","query":"vacation","retrieval_setting":{"top_k":1,"score_threshold":0}}')
        assert result['records'][0]['metadata'] == {'public': 'visible'}
        assert result['records'][0]['title'] == 'source'
    finally:
        app.close()


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'https://user:password@example.com', 'http://localhost?token=x', 'https://localhost/#x', 'localhost:1234'])
def test_url_boundary(url):
    with pytest.raises(ConfigError):
        http_url(url)


@pytest.mark.parametrize('key', [None, '', 'a b', 'key\nsecret'])
def test_key_required(key, json_app):
    with pytest.raises(ValueError):
        Application(json_app.config, key)
