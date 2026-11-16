import json
from pathlib import Path

import pytest

from ragwrench.lint.io import atomic_write, load_chunks, load_config
from ragwrench.lint.model import InputError


def write(tmp_path, text):
    path = tmp_path / 'chunks.jsonl'
    path.write_text(text)
    return path


def test_generic_and_metadata(tmp_path):
    record = {'id': 42, 'text': 'Hello.', 'metadata': {'nested': {'value': [1, 2]}}}
    chunk = load_chunks(write(tmp_path, json.dumps(record)))[0]
    assert chunk.id == '42'
    assert chunk.record() == {**record, 'id': '42'}


def test_langchain_constructor_is_data_only(tmp_path):
    record = {'lc': 1, 'type': 'constructor', 'id': ['dangerous', 'Class'],
              'kwargs': {'page_content': 'Safe text.', 'metadata': {'path': '../../outside'}}}
    chunk = load_chunks(write(tmp_path, json.dumps(record)))[0]
    assert chunk.text == 'Safe text.'
    assert chunk.metadata['path'] == '../../outside'


def test_llamaindex_json_array(tmp_path):
    record = {'id_': 'node-1', 'text': 'Hello.', 'start_char_idx': 3, 'end_char_idx': 9,
              'metadata': {'source': 'synthetic.md'}}
    chunk = load_chunks(write(tmp_path, json.dumps([record])), 'llamaindex')[0]
    assert (chunk.id, chunk.start_hint, chunk.end_hint) == ('node-1', 3, 9)


@pytest.mark.parametrize('raw,message', [
    ('{"text":"OK"}\n{"text":', 'line 2'),
    ('', 'empty chunk set'), ('[]', 'empty chunk set'),
    ('{"text":""}', 'non-empty string'), ('{"text":null}', 'non-empty string'),
    ('{"text":"x","metadata":[]}', 'metadata'), ('null', 'object'),
    ('{"text":"x","id":true}', 'id'),
    ('{"text":"x","id":"a"}\n{"text":"y","id":"a"}', 'duplicate chunk id'),
    ('{"text":"x","start_char_idx":0}', 'together'),
    ('{"text":"x","start_char_idx":0,"end_char_idx":-1}', 'hints'),
    ('{"text":"x","start_char_idx":true,"end_char_idx":4}', 'hints'),
    ('{"text":"x","metadata":{"number":NaN}}', 'constant'),
    ('{"text":"x","text":"y"}', 'duplicate JSON key'),
    ('{"type":"evil","kwargs":{}}', 'unsupported LangChain'),
])
def test_input_errors(tmp_path, raw, message):
    with pytest.raises(InputError, match=message):
        load_chunks(write(tmp_path, raw))


def test_crlf_jsonl(tmp_path):
    path = write(tmp_path, '{"text":"a"}\r\n{"text":"b"}\r\n')
    assert len(load_chunks(path)) == 2


def test_config_and_unknown_key(tmp_path):
    path = tmp_path / 'settings.toml'
    path.write_text('[chunklint]\nmin_chars=0\nmax_chars=100\ndisable=["CL005"]\n')
    config = load_config(path)
    assert (config.min_chars, config.max_chars, config.disable) == (0, 100, ['CL005'])
    path.write_text('[chunklint]\nexecute="rm -rf /"\n')
    with pytest.raises(InputError, match='unknown config'):
        load_config(path)
    path.write_text('[other]\nx=1\n')
    with pytest.raises(InputError, match='only a'):
        load_config(path)
    path.write_text('[chunklint\n')
    with pytest.raises(InputError, match='invalid TOML'):
        load_config(path)


def test_explicit_missing_config(tmp_path):
    with pytest.raises(InputError, match='does not exist'):
        load_config(tmp_path / 'missing.toml')


def test_atomic_output_refuses_symlinks(tmp_path):
    target = tmp_path / 'target'
    target.write_text('keep')
    link = tmp_path / 'link'
    link.symlink_to(target)
    with pytest.raises(InputError, match='symlink'):
        atomic_write(link, 'bad')
    assert target.read_text() == 'keep'


def test_input_limits(tmp_path, monkeypatch):
    import ragwrench.lint.io as module
    path = write(tmp_path, '{"text":"OK"}')
    monkeypatch.setattr(module, 'MAX_BYTES', 2)
    with pytest.raises(InputError, match='byte input limit'):
        load_chunks(path)
    monkeypatch.setattr(module, 'MAX_BYTES', 10000)
    monkeypatch.setattr(module, 'MAX_CHUNKS', 1)
    with pytest.raises(InputError, match='chunk limit'):
        load_chunks(write(tmp_path, '{"text":"a"}\n{"text":"b"}'))


def test_reject_special_input(tmp_path):
    import os
    path = tmp_path / 'pipe'
    os.mkfifo(path)
    with pytest.raises(InputError, match='regular file'):
        load_chunks(path)


@pytest.mark.parametrize('raw', [
    r'{"text":"hello","metadata":{"number":1e999}}',
    r'{"text":"\ud800"}',
    r'{"text":"hello","metadata":{"bad":"\ud800"}}',
])
def test_nonserializable_json_rejected_at_input_line(tmp_path, raw):
    with pytest.raises(InputError, match='line 1'):
        load_chunks(write(tmp_path, raw))


def test_langchain_nullable_id_is_optional(tmp_path):
    records = [{'id': None, 'page_content': 'A document.', 'metadata': {}, 'type': 'Document'},
               {'lc': 1, 'type': 'constructor', 'kwargs': {'id': None, 'page_content': 'Another.'}}]
    chunks = load_chunks(write(tmp_path, '\n'.join(json.dumps(record) for record in records)))
    assert [chunk.id for chunk in chunks] == ['c1', 'c2']


def test_unicode_separator_inside_json_string_roundtrips(tmp_path):
    text = 'Literal\u2028separator\u0085inside\u2029text.'
    chunk = load_chunks(write(tmp_path, json.dumps({'text': text}, ensure_ascii=False)))[0]
    assert chunk.text == text
