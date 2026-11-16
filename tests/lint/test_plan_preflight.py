"""Saved-plan preflight contract: plan diagnostics precede snapshot I/O.

Input: a saved plan with replacement[0].text = false, missing chunks/source,
and an existing binary output sentinel. Preview/apply must exit 2 with
$.edits[0].replacement[0].text, read neither snapshot, and write nothing.
Valid structure must still reach snapshot and deterministic repair validation;
the valid/apply and stale-snapshot contracts also run in tests/test_plan.py.
"""
import json
import subprocess
import sys

import pytest

from ragwrench.lint.model import Config
from ragwrench.lint.plan import create_plan


@pytest.fixture
def snapshots(tmp_path):
    chunks, source, plan, output = [tmp_path / name for name in
                                    ('input.json', 'source.md', 'plan.json', 'out.jsonl')]
    chunks.write_bytes('[{"id":"h","text":"# Café"},{"id":"p","text":"Body."}]'.encode())
    source.write_bytes('# Café\nBody.'.encode())
    value, _ = create_plan(chunks.read_bytes(), source.read_bytes(), config=Config(min_chars=0))
    assert len(value['edits']) == 1
    plan.write_text(json.dumps(value), encoding='utf-8')
    output.write_bytes(b'KEEP\x00\xff\r\n')
    return chunks, source, plan, output


def malformed(plan):
    value = json.loads(plan.read_text())
    value['edits'][0]['replacement'][0]['text'] = False
    plan.write_text(json.dumps(value), encoding='utf-8')


def args_for(command, chunks, source, plan, output, format='text'):
    args = [command, str(chunks), '--source', str(source), '--plan', str(plan), '--format', format]
    return args + (['-o', str(output)] if command == 'apply' else [])


@pytest.mark.parametrize('command', ['preview', 'apply'])
@pytest.mark.parametrize('missing', ['input', 'source'])
@pytest.mark.parametrize('format', ['text', 'json'])
def test_malformed_plan_precedes_unavailable_snapshot(snapshots, tmp_path, command, missing, format):
    chunks, source, plan, output = snapshots
    malformed(plan)
    (chunks if missing == 'input' else source).unlink()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = subprocess.run([sys.executable, '-m', 'ragwrench.lint',
                             *args_for(command, chunks, source, plan, output, format)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert not result.stdout
    assert result.stderr == ('ragwrench lint: invalid fix plan at '
                             '$.edits[0].replacement[0].text: expected string\n')
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('command', ['preview', 'apply'])
def test_malformed_plan_never_reads_snapshots_or_writes(snapshots, tmp_path, monkeypatch, capsys, command):
    import ragwrench.lint.cli as cli
    import ragwrench.lint.plan as plans

    chunks, source, plan, output = snapshots
    malformed(plan)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail('malformed plan attempted snapshot I/O, repair or a write')

    # load_plan uses io.read_text; CLI read_bytes is exclusively for snapshots.
    for name in ('read_bytes', 'atomic_write', 'validate_plan'):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(plans, 'create_plan', forbidden)
    assert cli.main(args_for(command, chunks, source, plan, output)) == 2
    captured = capsys.readouterr()
    assert '$.edits[0].replacement[0].text' in captured.err
    assert not captured.out
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('command', ['preview', 'apply'])
@pytest.mark.parametrize('missing', ['input', 'source'])
def test_valid_structure_proceeds_to_snapshot_reads(snapshots, tmp_path, capsys, command, missing):
    from ragwrench.lint.cli import main

    chunks, source, plan, output = snapshots
    unavailable = chunks if missing == 'input' else source
    unavailable.unlink()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert main(args_for(command, chunks, source, plan, output)) == 2
    captured = capsys.readouterr()
    assert f'{unavailable}: input must be a regular file' in captured.err
    assert not captured.out
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
