"""Synthetic integration contract for saved, snapshot-bound repair plans."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def run(*args, cwd):
    return subprocess.run([sys.executable, '-m', 'ragwrench.lint', *map(str, args)],
                          cwd=cwd, capture_output=True, text=True, timeout=20)


def canonical(records):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                              allow_nan=False) + '\n' for r in records).encode('utf-8')


@pytest.fixture
def saved_plan(tmp_path):
    records = [{'id': cid, 'text': text, 'metadata': {}} for cid, text in
               [('intro', 'Préface.'), ('h', '# Café'), ('p', 'Body.'), ('tail', 'Tail.')]]
    chunks, source, plan = [tmp_path / name for name in ('chunks.jsonl', 'source.md', 'plan.json')]
    # Raw snapshot hashing includes BOM and CRLF; byte coordinates use canonical JSONL.
    chunks.write_bytes(b'\xef\xbb\xbf' + b'\r\n'.join(json.dumps(r, ensure_ascii=False).encode() for r in records))
    source.write_bytes('Préface.\r\n# Café\r\nBody.\r\nTail.'.encode())
    result = run('plan', chunks, '--source', source, '--min-chars', 0, '-o', plan, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    return chunks, source, plan, records


def test_plan_preview_apply_exact_chunk_and_utf8_byte_edits(saved_plan, tmp_path):
    chunks, source, plan_path, records = saved_plan
    plan = json.loads(plan_path.read_text())
    assert plan['format'] == 'chunklint.fix-plan'
    assert plan['version'] == 1
    assert plan['input']['sha256'] == hashlib.sha256(chunks.read_bytes()).hexdigest()
    assert plan['source']['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(plan['edits']) == 1
    edit = plan['edits'][0]
    assert (edit['start_chunk'], edit['end_chunk']) == (1, 3)
    assert edit['start_byte'] == len(canonical(records[:1]))
    assert edit['end_byte'] == len(canonical(records[:3]))
    assert edit['before_sha256'] == hashlib.sha256(canonical(records[1:3])).hexdigest()
    assert edit['replacement'][0]['text'] == '# Café\r\nBody.'
    assert edit['replacement'][0]['id'] == 'h'
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    preview = run('preview', chunks, '--source', source, '--plan', plan_path, cwd=tmp_path)
    assert preview.returncode == 0, preview.stderr
    assert 'UTF-8 bytes' in preview.stdout
    assert '# Café' in preview.stdout and 'Body.' in preview.stdout
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    output = tmp_path / 'fixed.jsonl'
    applied = run('apply', chunks, '--source', source, '--plan', plan_path, '-o', output,
                  '--format', 'json', cwd=tmp_path)
    assert applied.returncode == 0, applied.stderr
    assert json.loads(applied.stdout)['invariant']['source_coverage_preserved']
    expected = canonical(records[:1] + edit['replacement'] + records[3:])
    assert output.read_bytes() == expected
    assert hashlib.sha256(expected).hexdigest() == plan['output_sha256']
    assert not Path(str(output) + '.changes.json').exists()
    assert all((tmp_path / name).read_bytes() == raw for name, raw in before.items())


@pytest.mark.parametrize('target', ['text', 'metadata', 'serialization', 'source', 'missing_source'])
def test_stale_snapshot_fails_before_any_write(saved_plan, tmp_path, target):
    chunks, source, plan, records = saved_plan
    if target in ('text', 'metadata'):
        if target == 'text':
            records[-1]['text'] = 'Changed tail.'
        else:
            records[-1]['metadata'] = {'new': True}
        chunks.write_bytes(canonical(records))
    elif target == 'serialization':
        chunks.write_bytes(chunks.read_bytes() + b'\n')
    elif target == 'source':
        source.write_bytes(source.read_bytes().replace(b'\r\n', b'\n'))
    options = [] if target == 'missing_source' else ['--source', source]
    output = tmp_path / 'fixed.jsonl'
    output.write_bytes(b'KEEP EXISTING OUTPUT')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for command in ('preview', 'apply'):
        write_args = ['-o', output] if command == 'apply' else []
        result = run(command, chunks, '--plan', plan, *options, *write_args, cwd=tmp_path)
        assert result.returncode == 2
        assert 'snapshot' in result.stderr
        assert not result.stdout
        assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('change', ['offset', 'float_offset', 'replacement', 'hash', 'version', 'config', 'extra'])
def test_invalid_plan_is_rejected_atomically(saved_plan, tmp_path, change):
    chunks, source, plan_path, _ = saved_plan
    plan = json.loads(plan_path.read_text())
    if change == 'offset':
        plan['edits'][0]['end_byte'] -= 1
    elif change == 'float_offset':
        plan['edits'][0]['start_chunk'] = float(plan['edits'][0]['start_chunk'])
    elif change == 'replacement':
        plan['edits'][0]['replacement'][0]['text'] = 'Lost payload'
    elif change == 'hash':
        plan['output_sha256'] = '0' * 64
    elif change == 'version':
        plan['version'] = 999
    elif change == 'config':
        plan['config']['disable'] = {}
    else:
        plan['execute'] = 'never execute data'
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / 'fixed.jsonl'
    before = set(tmp_path.iterdir())
    result = run('apply', chunks, '--source', source, '--plan', plan_path, '-o', output, cwd=tmp_path)
    assert result.returncode == 2
    assert 'Traceback' not in result.stderr
    assert not output.exists()
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize('adapter,filename', [
    ('auto', 'chunks_langchain.jsonl'), ('langchain', 'chunks_langchain.jsonl'),
    ('llamaindex', 'chunks_llamaindex.json'),
])
def test_plan_all_fixture_repairs_and_determinism(tmp_path, adapter, filename):
    from ragwrench.lint.fix import fix
    from ragwrench.lint.io import load_chunks, read_text
    from ragwrench.lint.model import Config

    examples = Path(__file__).resolve().parents[2] / 'examples' / 'lint'
    chunks, source = examples / filename, examples / 'doc.md'
    plan, second, output = [tmp_path / name for name in ('plan.json', 'second.json', 'out.jsonl')]
    options = ['--source', source, '--min-chars', 0, '--max-chars', 160, '--adapter', adapter]
    assert run('plan', chunks, *options, '-o', plan, cwd=tmp_path).returncode == 0
    assert run('plan', chunks, *options, '-o', second, cwd=tmp_path).returncode == 0
    assert plan.read_bytes() == second.read_bytes()
    result = run('apply', chunks, '--source', source, '--plan', plan, '-o', output, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    expected = fix(load_chunks(chunks, adapter), read_text(source), Config(min_chars=0, max_chars=160))
    assert output.read_bytes() == canonical([chunk.record() for chunk in expected.chunks])
    assert json.loads(plan.read_text())['edits']
    checked = run('check', output, '--source', source, '--min-chars', 0, '--max-chars', 160,
                  '--fail-on', 'warning', cwd=tmp_path)
    assert checked.returncode == 0, checked.stdout


def test_source_free_noop_plan_uses_saved_config(tmp_path):
    chunks, plan, output = [tmp_path / name for name in ('input.json', 'plan.json', 'out.jsonl')]
    records = [{'id': 'h', 'text': '# Title', 'metadata': {}},
               {'id': 'p', 'text': 'Body.', 'metadata': {}}]
    chunks.write_text(json.dumps(records))
    assert run('plan', chunks, '--disable', 'CL003', '--min-chars', 0, '-o', plan, cwd=tmp_path).returncode == 0
    data = json.loads(plan.read_text())
    assert data['source'] is None and data['edits'] == []
    # A subsequently changed/broken ambient config must not affect a saved plan.
    (tmp_path / 'chunklint.toml').write_text('invalid config')
    result = run('apply', chunks, '--plan', plan, '-o', output, '--format', 'json', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['invariant']['source_coverage_checked'] is False
    assert output.read_bytes() == canonical(records)
    source = tmp_path / 'source.md'
    source.write_text('# Title\nBody.')
    assert run('apply', chunks, '--source', source, '--plan', plan, '-o', output, cwd=tmp_path).returncode == 2
    assert output.read_bytes() == canonical(records)


def test_disjoint_merge_and_delete_edits_use_original_coordinates(tmp_path):
    chunks, plan, output = [tmp_path / name for name in ('input.json', 'plan.json', 'out.jsonl')]
    records = [{'id': cid, 'text': text, 'metadata': {}} for cid, text in
               [('h', '# Title'), ('p', 'Body.'), ('middle', 'Unchanged.'),
                ('copy', 'Unchanged.'), ('last', 'Finish.')]]
    chunks.write_text(json.dumps(records))
    assert run('plan', chunks, '--min-chars', 0, '-o', plan, cwd=tmp_path).returncode == 0
    edits = json.loads(plan.read_text())['edits']
    assert [(e['start_chunk'], e['end_chunk']) for e in edits] == [(0, 2), (3, 4)]
    assert edits[1]['replacement'] == []
    assert edits[1]['start_byte'] == len(canonical(records[:3]))
    result = run('apply', chunks, '--plan', plan, '-o', output, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    expected = canonical(edits[0]['replacement'] + records[2:3] + records[4:])
    assert output.read_bytes() == expected
    # A stale record in the later edit must not leave the earlier merge applied.
    records[3]['text'] = 'New text.'
    chunks.write_text(json.dumps(records))
    assert run('apply', chunks, '--plan', plan, '-o', output, cwd=tmp_path).returncode == 2
    assert output.read_bytes() == expected


@pytest.mark.parametrize('target', ['input', 'source', 'plan', 'config', 'symlink', 'missing_parent'])
def test_plan_and_apply_output_protection(saved_plan, tmp_path, target):
    chunks, source, plan, _ = saved_plan
    config = tmp_path / 'chunklint.toml'
    config.write_text('[chunklint]\nmin_chars=0\n')
    output = {'input': chunks, 'source': source, 'plan': plan, 'config': config,
              'symlink': tmp_path / 'link', 'missing_parent': tmp_path / 'absent' / 'out'}[target]
    if target == 'symlink':
        output.symlink_to(chunks)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    applied = run('apply', chunks, '--source', source, '--plan', plan, '-o', output, cwd=tmp_path)
    assert applied.returncode == 2
    # Plan creation protects input/source/config as well; replacing its own existing
    # plan is explicitly allowed because it is the named output, not a read input.
    if target != 'plan':
        planned = run('plan', chunks, '--source', source, '-o', output, cwd=tmp_path)
        assert planned.returncode == 2
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('raw', ['[]', '{', '{"format":"chunklint.fix-plan","version":true}',
                                 '{"format":"chunklint.fix-plan","version":1,"version":1}',
                                 '{"format":"chunklint.fix-plan","version":1,"config":NaN}'])
def test_malformed_plans_have_clean_errors(tmp_path, raw):
    chunks, plan, output = [tmp_path / name for name in ('input.json', 'plan.json', 'out.jsonl')]
    chunks.write_text('{"text":"Valid."}')
    plan.write_text(raw)
    result = run('apply', chunks, '--plan', plan, '-o', output, cwd=tmp_path)
    assert result.returncode == 2
    assert 'Traceback' not in result.stderr
    assert not output.exists()


def test_plan_thresholds_describe_after_report(tmp_path):
    chunks, plan, output = [tmp_path / name for name in ('input.json', 'plan.json', 'out.jsonl')]
    chunks.write_text('{"text":"# EOF heading"}')
    result = run('plan', chunks, '--fail-on', 'warning', '--min-chars', 0, '-o', plan, cwd=tmp_path)
    assert result.returncode == 1 and plan.exists()
    assert run('preview', chunks, '--plan', plan, cwd=tmp_path).returncode == 1
    applied = run('apply', chunks, '--plan', plan, '-o', output, cwd=tmp_path)
    assert applied.returncode == 1 and output.exists()
    assert 'CL003' in applied.stdout


def test_oversized_generated_plan_fails_before_write(tmp_path, monkeypatch, capsys):
    from ragwrench.lint.cli import main
    import ragwrench.lint.cli as cli

    chunks, plan = tmp_path / 'input.json', tmp_path / 'plan.json'
    chunks.write_text('{"text":"Valid."}')
    monkeypatch.setattr(cli, 'MAX_BYTES', 100)
    assert main(['plan', str(chunks), '-o', str(plan)]) == 2
    assert 'fix plan exceeds' in capsys.readouterr().err
    assert not plan.exists()
