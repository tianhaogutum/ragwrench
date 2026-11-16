"""Offline saved-plan validation contracts, independent of input/source snapshots."""
from copy import deepcopy
import json
from importlib.resources import files
from pathlib import Path
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from ragwrench.lint.model import Config, InputError
from ragwrench.lint.plan import create_plan, load_plan


def run(*args, cwd):
    return subprocess.run([sys.executable, '-m', 'ragwrench.lint', *map(str, args)],
                          cwd=cwd, capture_output=True, text=True, timeout=20)


@pytest.fixture
def plan():
    raw = json.dumps([{'id': 'h', 'text': '# Café'}, {'id': 'p', 'text': 'Body.'}]).encode()
    value, _ = create_plan(raw, '# Café\nBody.'.encode(), config=Config(min_chars=0))
    assert len(value['edits']) == 1
    return value


def save(tmp_path, plan):
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan, ensure_ascii=False), encoding='utf-8')
    return path


def test_standalone_validation_needs_only_plan(plan, tmp_path):
    path = save(tmp_path, plan)
    # No original input/source files exist; even malformed ambient config is ignored.
    (tmp_path / 'chunklint.toml').write_text('broken config')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = run('validate-plan', path, '--format', 'json', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'valid': True, 'version': 1, 'edits': 1,
                                       'snapshots_checked': False}
    assert not result.stderr
    text = run('validate-plan', path, cwd=tmp_path)
    assert text.returncode == 0, text.stderr
    assert text.stdout == 'Valid fix plan v1: 1 edits (snapshots not checked)\n'
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_standalone_does_not_load_snapshots_or_run_repairs(plan, tmp_path, monkeypatch, capsys):
    from ragwrench.lint.cli import main
    import ragwrench.lint.cli as cli
    import ragwrench.lint.plan as plans

    def forbidden(*args, **kwargs):
        pytest.fail('standalone validation accessed chunks/config/repair')

    path = save(tmp_path, plan)
    for name in ('load_chunks', 'load_config', 'read_bytes', 'create_plan', 'validate_plan'):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(plans, 'create_plan', forbidden)
    assert main(['validate-plan', str(path)]) == 0
    assert 'snapshots not checked' in capsys.readouterr().out


# Every mutation below is invalid under the published schema itself.
SCHEMA_CASES = [
    ('format', 'unknown'), ('version', True), ('version', 2),
    ('tool_version', ''), ('coordinates', 'characters'),
    ('input', []), ('input.sha256', 'A' * 64), ('input.sha256', '0' * 63),
    ('input.sha256', 'a' * 64 + '\n'), ('input.bytes', -1), ('input.bytes', True), ('input.adapter', 'executable'),
    ('input.path', '/never/read'), ('source', {}), ('source.bytes', 1.5),
    ('source.path', '/never/read'), ('output_sha256', 'bad'),
    ('config.min_chars', -1), ('config.max_chars', 0), ('config.near_duplicate', 0),
    ('config.max_overlap', 1.1), ('config.disable', ['CL999']),
    ('config.fail_on', []), ('config.max_pair_checks', False), ('config.extra', 0),
    ('edits', {}), ('edits.0.start_chunk', -1), ('edits.0.end_byte', 1.5),
    ('edits.0.before_sha256', 'xyz'), ('edits.0.extra', 0),
    ('edits.0.replacement', {}), ('edits.0.replacement.0.id', ''),
    ('edits.0.replacement.0.text', '\t\n '), ('edits.0.replacement.0.metadata', []),
    ('edits.0.replacement.0.execute', 'inert'),
    ('edits.0.replacement.0.start_char_idx', True),
]


def assign(plan, dotted, value):
    parts = dotted.split('.')
    target = plan
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    key = int(parts[-1]) if isinstance(target, list) else parts[-1]
    target[key] = value


@pytest.mark.parametrize('field,value', SCHEMA_CASES)
def test_field_specific_structural_errors(plan, tmp_path, field, value):
    assign(plan, field, value)
    path = save(tmp_path, plan)
    with pytest.raises(InputError) as error:
        load_plan(path)
    # JSON paths identify the erroneous field, including indices and unknown keys.
    expected = '$.' + field.replace('.0', '[0]')
    assert expected in str(error.value)


def test_published_schema_matches_structural_contract(plan, tmp_path):
    result = run('validate-plan', '--schema', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    assert schema == json.loads(files('ragwrench.lint').joinpath('schemas', 'fix-plan-v1.schema.json').read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(plan)
    for field, value in SCHEMA_CASES:
        invalid = deepcopy(plan)
        assign(invalid, field, value)
        assert not validator.is_valid(invalid), field
    for key in plan:
        invalid = deepcopy(plan)
        del invalid[key]
        assert not validator.is_valid(invalid), key
        with pytest.raises(InputError, match=r'\$\.' + key):
            load_plan(save(tmp_path, invalid))


@pytest.mark.parametrize('case,field', [
    ('config_order', '$.config.max_chars'), ('chunk_range', '$.edits[0].end_chunk'),
    ('byte_range', '$.edits[0].end_byte'), ('overlap', '$.edits[1].start_chunk'),
    ('byte_overlap', '$.edits[1].start_byte'),
    ('unpaired_hint', '$.edits[0].replacement[0].end_char_idx'),
    ('reversed_hint', '$.edits[0].replacement[0].end_char_idx'),
    ('duplicate_id', '$.edits[0].replacement[1].id'),
    ('float_coordinate', '$.edits[0].start_chunk'),
    ('tool_version', '$.tool_version'),
])
def test_semantic_errors_without_snapshots(plan, tmp_path, case, field):
    edit = plan['edits'][0]
    if case == 'config_order':
        plan['config'].update(min_chars=10, max_chars=5)
    elif case == 'chunk_range':
        edit.update(start_chunk=3, end_chunk=2)
    elif case == 'byte_range':
        edit.update(start_byte=10, end_byte=5)
    elif case == 'overlap':
        plan['edits'].append(deepcopy(edit))
    elif case == 'byte_overlap':
        other = deepcopy(edit)
        other.update(start_chunk=edit['end_chunk'], end_chunk=edit['end_chunk'] + 1)
        plan['edits'].append(other)
    elif case == 'unpaired_hint':
        edit['replacement'][0].pop('end_char_idx')
    elif case == 'reversed_hint':
        edit['replacement'][0].update(start_char_idx=5, end_char_idx=4)
    elif case == 'duplicate_id':
        edit['replacement'].append(deepcopy(edit['replacement'][0]))
    elif case == 'float_coordinate':
        edit['start_chunk'] = 0.0
    else:
        plan['tool_version'] = '999.0'
    with pytest.raises(InputError) as error:
        load_plan(save(tmp_path, plan))
    assert field in str(error.value)


def test_json_and_text_negative_diagnostics(plan, tmp_path):
    plan['edits'][0]['replacement'][0]['text'] = False
    path = save(tmp_path, plan)
    before = path.read_bytes()
    result = run('validate-plan', path, '--format', 'json', cwd=tmp_path)
    assert result.returncode == 2 and not result.stderr
    data = json.loads(result.stdout)
    assert data['valid'] is False
    assert data['diagnostics'][0]['path'] == '$.edits[0].replacement[0].text'
    assert data['diagnostics'][0]['message']
    text = run('validate-plan', path, cwd=tmp_path)
    assert text.returncode == 2 and not text.stdout
    assert '$.edits[0].replacement[0].text' in text.stderr
    assert 'Traceback' not in text.stderr
    assert before == path.read_bytes()


@pytest.mark.parametrize('raw', ['{', '[]', '{"version":1,"version":1}',
                                 '{"value":NaN}', '{"value":1e999}', '{"value":"\\ud800"}'])
def test_malformed_json_has_root_diagnostic(tmp_path, raw):
    path = tmp_path / 'plan.json'
    path.write_text(raw)
    result = run('validate-plan', path, '--format', 'json', cwd=tmp_path)
    assert result.returncode == 2 and not result.stderr
    assert json.loads(result.stdout)['diagnostics'][0]['path'] == '$'


def test_missing_or_oversized_file_is_exit_two(tmp_path, monkeypatch, capsys):
    from ragwrench.lint.cli import main
    import ragwrench.lint.io as io
    path = tmp_path / 'missing.json'
    assert main(['validate-plan', str(path), '--format', 'json']) == 2
    assert json.loads(capsys.readouterr().out)['valid'] is False
    path.write_text(' ' * 20)
    monkeypatch.setattr(io, 'MAX_BYTES', 10)
    assert main(['validate-plan', str(path), '--format', 'json']) == 2
    assert 'byte input limit' in json.loads(capsys.readouterr().out)['diagnostics'][0]['message']


def test_structurally_valid_hash_tampering_still_fails_apply(plan, tmp_path):
    # Standalone validation cannot assert freshness or hashes without snapshots.
    plan['output_sha256'] = '0' * 64
    path = save(tmp_path, plan)
    assert run('validate-plan', path, cwd=tmp_path).returncode == 0
    chunks, source, output = [tmp_path / name for name in ('input.json', 'source.md', 'out.jsonl')]
    chunks.write_bytes(json.dumps([{'id': 'h', 'text': '# Café'}, {'id': 'p', 'text': 'Body.'}]).encode())
    source.write_bytes('# Café\nBody.'.encode())
    output.write_bytes(b'KEEP')
    applied = run('apply', chunks, '--source', source, '--plan', path, '-o', output, cwd=tmp_path)
    assert applied.returncode == 2 and output.read_bytes() == b'KEEP'


def test_source_free_noop_plan_and_inert_metadata(tmp_path):
    raw = json.dumps([{'text': 'Valid.', 'metadata': {'execute': '$(never)', 'path': '../inert'}}]).encode()
    plan, _ = create_plan(raw, config=Config(min_chars=0))
    assert plan['source'] is None and plan['edits'] == []
    result = run('validate-plan', save(tmp_path, plan), '--format', 'json', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['edits'] == 0


@pytest.mark.parametrize('args', [[], ['--schema', 'plan.json'], ['--schema', '--format', 'json']])
def test_validate_plan_requires_plan_xor_schema(tmp_path, args):
    result = run('validate-plan', *args, cwd=tmp_path)
    assert result.returncode == 2
    assert 'Traceback' not in result.stderr


def test_validation_exit_is_independent_of_lint_threshold(tmp_path):
    plan, result = create_plan(b'{"text":"```python\\nunclosed"}',
                               config=Config(min_chars=0, fail_on='warning'))
    assert any(issue.rule == 'CL001' for issue in result.after.issues)
    result = run('validate-plan', save(tmp_path, plan), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
