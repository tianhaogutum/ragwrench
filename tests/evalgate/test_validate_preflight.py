"""Single archived audit CLI contract, established before runtime edits."""
import json
from pathlib import Path

import pytest

from ragwrench.evalgate.cli import main
from .test_diff_preflight import audit, forbid_work


RECEIPT = '{\n  "schema_version": 1,\n  "command": "validate-preflight",\n  "valid": true\n}\n'


def args(path, *extra):
    return ['validate-preflight', '--input', str(path), *map(str, extra)]


@pytest.fixture
def archived(tmp_path):
    path = tmp_path / 'archived.json'
    path.write_text(json.dumps(audit()))
    return path


def standalone(monkeypatch):
    forbid_work(monkeypatch)
    def forbidden(*a, **kw):
        pytest.fail('single audit validation must not invoke diff or compare')
    for name in ('diff_preflight', 'compare', 'compare_many', 'compare_saved', 'compare_many_saved'):
        monkeypatch.setattr(f'ragwrench.evalgate.cli.{name}', forbidden)


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('method', ['bootstrap', 'randomization', 'ttest'])
def test_valid_receipt_reads_only_one_audit(archived, saved, method, monkeypatch, capsys):
    archived.write_text(json.dumps(audit(saved=saved, method=method)))
    before = archived.read_bytes()
    standalone(monkeypatch)
    reads = []
    read_text = Path.read_text
    def read(path, *a, **kw):
        reads.append(path)
        return read_text(path, *a, **kw)
    monkeypatch.setattr(Path, 'read_text', read)
    assert main(args(archived)) == 0
    captured = capsys.readouterr()
    assert captured.out == RECEIPT and captured.err == ''
    assert reads == [archived] and archived.read_bytes() == before


def test_deterministic_output_and_force(archived, monkeypatch, capsys):
    standalone(monkeypatch)
    output = archived.with_name('receipt.json')
    assert main(args(archived, '--output', output)) == 0
    captured = capsys.readouterr()
    assert captured.out == captured.err == '' and output.read_text() == RECEIPT
    output.write_bytes(b'old receipt\n')
    assert main(args(archived, '--output', output)) == 2
    assert capsys.readouterr().out == '' and output.read_bytes() == b'old receipt\n'
    assert main(args(archived, '--output', output, '--force')) == 0
    assert capsys.readouterr().out == '' and output.read_text() == RECEIPT
    # Input key ordering/whitespace and --force without an output do not affect bytes.
    value = json.loads(archived.read_text())
    archived.write_text(json.dumps(dict(reversed(list(value.items()))), indent=4))
    assert main(args(archived, '--force')) == 0
    assert capsys.readouterr().out == RECEIPT
    assert not list(output.parent.glob('.retgate-*'))


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('bad', ['malformed', 'duplicate', 'nested_duplicate', 'nonfinite', 'overflow',
                                 'null', 'array', 'unbound', 'version', 'unknown', 'count', 'work',
                                 'policy', 'late_hash', 'late_field', 'missing_binding', 'encoding',
                                 'missing', 'directory', 'trailing'])
def test_invalid_no_receipt_preserves_outputs(archived, bad, existing, monkeypatch, capsys):
    value = json.loads(archived.read_text())
    raw = {'malformed': '{', 'duplicate': '{"x":1,"x":2}', 'nonfinite': 'NaN',
           'null': 'null', 'array': '[]', 'trailing': json.dumps(value) + ' {}'}
    if bad in raw:
        archived.write_text(raw[bad])
    elif bad == 'nested_duplicate':
        archived.write_text(json.dumps(value).replace('"seed": 42', '"seed": 42, "seed": 42'))
    elif bad == 'overflow':
        archived.write_text(json.dumps(value).replace('"confidence": 0.95', '"confidence": 1e999'))
    elif bad == 'encoding':
        archived.write_bytes(b'\xff')
    elif bad == 'missing':
        archived.unlink()
    elif bad == 'directory':
        archived.unlink()
        archived.mkdir()
    else:
        if bad == 'unbound':
            del value['input_bindings']
        elif bad == 'version':
            value['schema_version'] = 2
        elif bad == 'unknown':
            value['unknown'] = True
        elif bad == 'count':
            value['family_size'] += 1
        elif bad == 'work':
            value['estimated_family_resamples'] += 1
        elif bad == 'policy':
            value['policy']['effective_thresholds']['z']['hit@1'] = .5
        elif bad == 'late_hash':
            value['input_bindings']['candidates']['z']['run_sha256'] = 'A' * 64
        elif bad == 'late_field':
            value['input_bindings']['candidates']['z']['extra'] = 0
        elif bad == 'missing_binding':
            del value['input_bindings']['candidates']['z']
        archived.write_text(json.dumps(value))
    output = archived.with_name('protected.json')
    if existing:
        output.write_bytes(b'sentinel\n')
    standalone(monkeypatch)
    assert main(args(archived, '--output', output, '--force')) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'audit' in captured.err
    assert output.read_bytes() == b'sentinel\n' if existing else not output.exists()
    assert not list(output.parent.glob('.retgate-*'))


@pytest.mark.parametrize('alias', ['direct', 'resolved', 'symlink', 'hardlink', 'input_symlink', 'dangling'])
def test_input_aliases_and_symlink_outputs_protected(archived, alias, monkeypatch, capsys):
    before = archived.read_bytes()
    output = archived.with_name('alias.json')
    input_path = archived
    if alias == 'direct':
        output = archived
    elif alias == 'resolved':
        sub = archived.parent / 'sub'
        sub.mkdir()
        output = sub / '..' / archived.name
    elif alias in ('symlink', 'dangling'):
        output.symlink_to(archived if alias == 'symlink' else archived.with_name('absent'))
    elif alias == 'hardlink':
        output.hardlink_to(archived)
    elif alias == 'input_symlink':
        output = archived
        input_path = archived.with_name('input-symlink.json')
        input_path.symlink_to(archived)
    standalone(monkeypatch)
    assert main(args(input_path, '--output', output, '--force')) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'overwrite an input' in captured.err
    assert archived.read_bytes() == before
    if alias in ('symlink', 'dangling'):
        assert output.is_symlink()
    assert not list(output.parent.glob('.retgate-*'))


@pytest.mark.parametrize('kind', ['parent', 'directory', 'write', 'flush', 'fsync', 'link', 'replace'])
def test_failed_output_preserves_destination(archived, kind, monkeypatch, capsys):
    import ragwrench.evalgate.cli as cli
    output = archived.with_name('protected.json')
    output.write_bytes(b'sentinel\n')
    standalone(monkeypatch)
    extra = ['--force']
    if kind == 'parent':
        output = archived.parent / 'missing' / 'out.json'
    elif kind == 'directory':
        output.unlink()
        output.mkdir()
    else:
        def fail(*a, **kw):
            raise OSError('simulated ' + kind + ' failure')
        if kind in ('write', 'flush'):
            real = cli.tempfile.NamedTemporaryFile
            class FailingStream:
                def __enter__(self):
                    self.stream = real(mode='w', encoding='utf-8', dir=output.parent,
                                       prefix='.retgate-', delete=False)
                    self.name = self.stream.name
                    return self
                def __getattr__(self, name):
                    return fail if name == kind else getattr(self.stream, name)
                def __exit__(self, *a):
                    self.stream.close()
            monkeypatch.setattr(cli.tempfile, 'NamedTemporaryFile', lambda **kw: FailingStream())
        else:
            monkeypatch.setattr(cli.os, kind, fail)
            if kind == 'link':
                output.unlink()
                extra = []
    assert main(args(archived, '--output', output, *extra)) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'error:' in captured.err
    if kind in ('parent', 'link'):
        assert not output.exists()
    elif kind == 'directory':
        assert output.is_dir()
    else:
        assert output.read_bytes() == b'sentinel\n'
    assert not list(archived.parent.glob('.retgate-*'))


def test_uses_existing_mapping_validator(archived, monkeypatch, capsys):
    import ragwrench.evalgate.cli as cli
    from ragwrench.evalgate.preflight import validate_preflight
    calls = []
    def validate(value):
        calls.append(value)
        return validate_preflight(value)
    monkeypatch.setattr(cli, 'validate_preflight', validate)
    assert main(args(archived)) == 0
    assert len(calls) == 1 and calls[0] == json.loads(archived.read_text())
    assert capsys.readouterr().out == RECEIPT


@pytest.mark.parametrize('kind', ['input', 'output_parent'])
def test_symlink_loops_are_io_errors(archived, kind, monkeypatch, capsys):
    loop = archived.with_name('loop')
    loop.symlink_to(loop)
    output = archived.with_name('protected.json')
    output.write_bytes(b'sentinel\n')
    input_path = loop if kind == 'input' else archived
    destination = output if kind == 'input' else loop / 'receipt.json'
    standalone(monkeypatch)
    assert main(args(input_path, '--output', destination, '--force')) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'error:' in captured.err
    assert output.read_bytes() == b'sentinel\n' and loop.is_symlink()
    assert not list(output.parent.glob('.retgate-*'))
