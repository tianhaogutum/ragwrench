import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ragwrench.lint.bench import benchmark

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/lint/chunks_langchain.jsonl'
SOURCE = ROOT / 'examples/lint/doc.md'


def run(*args, cwd=ROOT):
    env = dict(os.environ, UV_OFFLINE='1', NO_PROXY='*', HTTP_PROXY='http://127.0.0.1:1',
               HTTPS_PROXY='http://127.0.0.1:1')
    return subprocess.run([sys.executable, '-m', 'ragwrench.lint', *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, env=env, timeout=20)


def fixture(tmp_path, text='A complete paragraph.'):
    path = tmp_path / 'chunks.jsonl'
    path.write_text(json.dumps({'id': 'chunk', 'text': text}) + '\n')
    return path


def test_real_format_end_to_end(tmp_path):
    before = run('check', EXAMPLE, '--source', SOURCE, '--format', 'json')
    assert before.returncode == 1
    assert json.loads(before.stdout)['stats']['source']['coverage'] == 1
    output = tmp_path / 'fixed.jsonl'
    repaired = run('fix', EXAMPLE, '--source', SOURCE, '--min-chars', 0, '--max-chars', 160, '-o', output)
    assert repaired.returncode == 0, repaired.stderr
    assert output.exists()
    log = json.loads(Path(str(output) + '.changes.json').read_text())
    assert len(log['changes']) == 5
    assert log['after']['stats']['defects'] == 0
    after = run('check', output, '--source', SOURCE, '--min-chars', 0, '--max-chars', 160)
    assert after.returncode == 0
    second = tmp_path / 'fixed-again.jsonl'
    assert run('fix', output, '--source', SOURCE, '--min-chars', 0, '--max-chars', 160, '-o', second).returncode == 0
    assert second.read_bytes() == output.read_bytes()


def test_real_splitter_quoted_fence_cli(tmp_path):
    fixture = ROOT / 'tests/lint/fixtures/quoted_fence'
    chunks, source = fixture / 'chunks.jsonl', fixture / 'source.md'
    before = run('check', chunks, '--source', source, '--min-chars', 0, '--format', 'json')
    assert before.returncode == 1
    assert any(issue['rule'] == 'CL001' for issue in json.loads(before.stdout)['issues'])
    output = tmp_path / 'quoted-fixed.jsonl'
    repaired = run('fix', chunks, '--source', source, '--min-chars', 0, '-o', output)
    assert repaired.returncode == 0, repaired.stderr
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]['text'] == source.read_text()
    assert len(records[0]['metadata']['chunklint']['inputs']) == 3
    after = run('check', output, '--source', source, '--min-chars', 0, '--fail-on', 'warning')
    assert after.returncode == 0, after.stdout


@pytest.mark.parametrize('adapter,path', [('auto', EXAMPLE), ('langchain', EXAMPLE),
                                          ('llamaindex', ROOT / 'examples/lint/chunks_llamaindex.json')])
def test_adapters_cli(adapter, path):
    result = run('check', path, '--source', SOURCE, '--adapter', adapter, '--format', 'json')
    assert result.returncode == 1
    assert json.loads(result.stdout)['stats']['source']['coverage'] == 1


def test_sarif_required_fields_and_locations():
    result = run('check', EXAMPLE, '--source', SOURCE, '--format', 'sarif')
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data['version'] == '2.1.0'
    assert data['$schema'].endswith('sarif-2.1.0.json')
    run_data = data['runs'][0]
    assert run_data['tool']['driver']['name'] == 'chunklint'
    rules = {r['id'] for r in run_data['tool']['driver']['rules']}
    assert len(rules) == 8
    for finding in run_data['results']:
        assert finding['ruleId'] in rules
        assert finding['level'] in ('note', 'warning', 'error')
        assert finding['message']['text']
        loc = finding['locations'][0]['physicalLocation']
        artifact = loc['artifactLocation']
        assert artifact['uri'] == run_data['artifacts'][artifact['index']]['location']['uri']
        if 'region' in loc:
            assert loc['region']['startLine'] >= 1


def test_sarif_without_source(tmp_path):
    path = fixture(tmp_path, '```')
    result = run('check', path, '--format', 'sarif')
    finding = json.loads(result.stdout)['runs'][0]['results'][0]
    assert finding['locations'][0]['physicalLocation']['artifactLocation']['index'] == 0


@pytest.mark.parametrize('fail_on,expected', [('error', 0), ('warning', 1), ('note', 1), ('none', 0)])
def test_severity_exit_codes(tmp_path, fail_on, expected):
    path = fixture(tmp_path, '# Orphan')
    result = run('check', path, '--fail-on', fail_on, cwd=tmp_path)
    assert result.returncode == expected
    assert 'CL003 warning' in result.stdout


def test_invalid_json_exit_and_line_number(tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text('{"text":"ok"}\n{bad}\n')
    result = run('check', path)
    assert result.returncode == 2
    assert 'line 2' in result.stderr
    assert not result.stdout


@pytest.mark.parametrize('raw', ['', '[]', '{}'])
def test_empty_and_missing_text_cli(tmp_path, raw):
    path = fixture(tmp_path)
    path.write_text(raw)
    result = run('check', path)
    assert result.returncode == 2
    assert 'Traceback' not in result.stderr


def test_mismatching_source_cli(tmp_path):
    path = fixture(tmp_path, 'not present')
    source = tmp_path / 'source.md'
    source.write_text('Unrelated original.')
    result = run('check', path, '--source', source)
    assert result.returncode == 1
    assert 'alignment failure rate: 100.00%' in result.stdout


def test_dry_run_writes_nothing(tmp_path):
    output = tmp_path / 'out.jsonl'
    log = tmp_path / 'log.json'
    before = set(tmp_path.iterdir())
    result = run('fix', EXAMPLE, '--source', SOURCE, '-o', output, '--change-log', log, '--dry-run', '--format', 'json')
    assert result.returncode == 0
    assert json.loads(result.stdout)['changes']
    assert set(tmp_path.iterdir()) == before


def test_dry_run_needs_no_output():
    assert run('fix', EXAMPLE, '--dry-run').returncode == 0


def test_fix_requires_output():
    result = run('fix', EXAMPLE)
    assert result.returncode == 2
    assert 'requires --output' in result.stderr


def test_refuse_input_overwrite(tmp_path):
    path = fixture(tmp_path, '# End')
    original = path.read_bytes()
    result = run('fix', path, '-o', path)
    assert result.returncode == 2
    assert path.read_bytes() == original


def test_refuse_source_overwrite(tmp_path):
    path = fixture(tmp_path)
    source = tmp_path / 'source.md'
    source.write_text('A complete paragraph.')
    result = run('fix', path, '--source', source, '-o', source)
    assert result.returncode == 2
    assert source.read_text() == 'A complete paragraph.'


def test_output_log_collision(tmp_path):
    path = fixture(tmp_path)
    output = tmp_path / 'out.jsonl'
    result = run('fix', path, '-o', output, '--change-log', output)
    assert result.returncode == 2
    assert not output.exists()


def test_refuse_symlink_and_missing_parent(tmp_path):
    path = fixture(tmp_path)
    output = tmp_path / 'out.jsonl'
    target = tmp_path / 'target'
    target.write_text('keep')
    output.symlink_to(target)
    assert run('fix', path, '-o', output).returncode == 2
    assert target.read_text() == 'keep'
    assert run('fix', path, '-o', tmp_path / 'absent/out').returncode == 2


def test_metadata_paths_and_expressions_are_inert(tmp_path):
    output = tmp_path / 'out.jsonl'
    marker = tmp_path / 'SHOULD_NOT_EXIST'
    text = f'$(touch {marker})'
    path = tmp_path / 'input.jsonl'
    path.write_text(json.dumps({'text': text, 'metadata': {'path': '../../escape', 'command': text}}))
    result = run('fix', path, '-o', output)
    assert result.returncode == 0
    assert not marker.exists()
    assert json.loads(output.read_text())['text'] == text
    assert sorted(p.name for p in tmp_path.iterdir()) == ['input.jsonl', 'out.jsonl', 'out.jsonl.changes.json']


def test_config_discovery_and_override(tmp_path):
    path = fixture(tmp_path, '# End')
    config = tmp_path / 'chunklint.toml'
    config.write_text('[chunklint]\nfail_on="warning"\ndisable=["CL005"]\n')
    assert run('check', path, cwd=tmp_path).returncode == 1
    assert run('check', path, '--disable', 'CL003', cwd=tmp_path).returncode == 0
    assert run('check', path, '--fail-on', 'none', cwd=tmp_path).returncode == 0
    assert run('check', path, '--disable', 'CL999', cwd=tmp_path).returncode == 2
    config.unlink()
    assert run('check', path, '--config', config, cwd=tmp_path).returncode == 2


def test_invalid_config_data_no_traceback(tmp_path):
    path = fixture(tmp_path)
    config = tmp_path / 'bad.toml'
    config.write_text('[chunklint]\ndisable=[{}]\n')
    result = run('check', path, '--config', config)
    assert result.returncode == 2
    assert 'Traceback' not in result.stderr


def test_bench_deterministic_and_recorded(tmp_path):
    first, second = run('bench'), run('bench')
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    result = json.loads(first.stdout)
    assert result == benchmark()
    assert result['after']['defects'] < result['before']['defects']
    assert result['invariant']['source_coverage_preserved']
    assert result['idempotent']
    path = tmp_path / 'bench.json'
    assert run('bench', '--write', path).returncode == 0
    assert path.read_text() == first.stdout
    committed = ROOT / 'docs/lint/benchmark.json'
    if committed.exists():
        assert json.loads(committed.read_text()) == result


def test_help_and_version():
    assert 'conservative repair' in run('--help').stdout
    assert run('--version').stdout == 'ragwrench lint 0.1.0\n'


def test_packaged_fixture_matches_example():
    from importlib.resources import files
    for name in ('doc.md', 'chunks_langchain.jsonl'):
        assert files('ragwrench.lint').joinpath('data', name).read_bytes() == (ROOT / 'examples' / 'lint' / name).read_bytes()


def test_failed_conservation_aborts_before_writing(tmp_path, monkeypatch, capsys):
    from ragwrench.lint.cli import main
    import ragwrench.lint.fix as engine
    path = fixture(tmp_path, '```py\nimportant_data = 1234\n```\n')
    output = tmp_path / 'fixed.jsonl'
    # Simulate a broken transformation: the public guard must block writing it.
    monkeypatch.setattr(engine, '_split_structures', lambda *args: ['lost', 'payload'])
    assert main(['fix', str(path), '-o', str(output)]) == 2
    assert 'conservation invariant failed' in capsys.readouterr().err
    assert not output.exists()
    assert not Path(str(output) + '.changes.json').exists()
