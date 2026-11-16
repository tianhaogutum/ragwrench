"""Single saved evaluation CLI contract, established before runtime edits."""
import json
from pathlib import Path

import pytest

from ragwrench.evalgate.cli import main
from ragwrench.evalgate.metrics import parse_metrics
from ragwrench.evalgate.saved import saved_evaluation


RECEIPT = '{\n  "schema_version": 1,\n  "command": "validate-evaluation",\n  "valid": true\n}\n'


def args(path, *extra):
    return ['validate-evaluation', '--input', str(path), *map(str, extra)]


def artifact(gain='linear', single=False):
    q = {'q1': {'d': 2}, 'q2': {'d': 1}, 'z': {'d': 1}}
    run = {'q1': {'d': 1.0}, 'q2': {'x': 2.0, 'd': 1.0}, 'extra': {'x': 1.0}}
    if single:
        q, run = {'q1': q['q1']}, {'q1': run['q1']}
    return saved_evaluation(q, run, parse_metrics('hit@1,ndcg@2,recall@2'), gain)


@pytest.fixture
def archived(tmp_path):
    path = tmp_path / 'archived.json'
    path.write_text(json.dumps(artifact()))
    return path


def standalone(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('single evaluation validation must not read retrieval inputs, score, compare or infer')
    for module, names in [
        ('ragwrench.evalgate.cli', ('read_qrels', 'read_run', 'saved_evaluation', 'validate_preflight',
                         'diff_preflight', 'compare', 'compare_many', 'compare_saved', 'compare_many_saved')),
        ('ragwrench.evalgate.formats', ('read_qrels', 'read_run')),
        ('ragwrench.evalgate.saved', ('evaluate', 'saved_evaluation', 'bind_evaluation')),
        ('ragwrench.evalgate.compare', ('evaluate', 'paired_test', 'adjust_p_values', 'bind_evaluation')),
        ('ragwrench.evalgate.metrics', ('evaluate',)),
        ('ragwrench.evalgate.statistics', ('paired_test', 'adjust_p_values')),
    ]:
        for name in names:
            monkeypatch.setattr(f'{module}.{name}', forbidden)


@pytest.mark.parametrize('gain', ['linear', 'exponential'])
@pytest.mark.parametrize('single', [False, True])
def test_valid_receipt_reads_only_one_artifact(archived, gain, single, monkeypatch, capsys):
    archived.write_text(json.dumps(artifact(gain, single)))
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


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('bad', ['malformed', 'duplicate', 'nested_duplicate', 'nan', 'infinity',
                                 'overflow', 'null', 'array', 'encoding', 'missing', 'directory', 'trailing'])
def test_strict_decoder_and_read_errors_preserve_output(archived, existing, bad, monkeypatch, capsys):
    value = json.loads(archived.read_text())
    raw = {'malformed': '{', 'duplicate': '{"x":1,"x":2}', 'nan': 'NaN', 'infinity': 'Infinity',
           'null': 'null', 'array': '[]', 'trailing': json.dumps(value) + ' {}',
           'nested_duplicate': json.dumps(value).replace('"gain": "linear"', '"gain": "linear", "gain": "linear"'),
           'overflow': json.dumps(value).replace('"hit@1": 0.0', '"hit@1": 1e999')}
    if bad in raw:
        archived.write_text(raw[bad])
    elif bad == 'encoding':
        archived.write_bytes(b'\xff')
    elif bad in ('missing', 'directory'):
        archived.unlink()
        if bad == 'directory':
            archived.mkdir()
    output = archived.with_name('protected.json')
    if existing:
        output.write_bytes(b'sentinel\n')
    standalone(monkeypatch)
    assert main(args(archived, '--output', output, '--force')) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'error:' in captured.err
    assert output.read_bytes() == b'sentinel\n' if existing else not output.exists()
    assert not list(output.parent.glob('.retgate-*'))


MUTATIONS = [
    ('unbound', lambda r: r.pop('manifest')),
    ('report_version', lambda r: r.update(schema_version=2)),
    ('manifest_version', lambda r: r['manifest'].update(schema_version=2)),
    ('evaluator_version', lambda r: r['manifest']['evaluator'].update(version='future')),
    ('unknown_field', lambda r: r.update(extra=True)),
    ('late_manifest_field', lambda r: r['manifest']['parameters'].update(extra=True)),
    ('late_missing_row', lambda r: r['per_query'].pop('z')),
    ('late_missing_metric', lambda r: r['per_query']['z'].pop('recall@2')),
    ('late_score_bool', lambda r: r['per_query']['z'].update({'recall@2': True})),
    ('late_score_string', lambda r: r['per_query']['z'].update({'recall@2': '0'})),
    ('late_score_negative', lambda r: r['per_query']['z'].update({'recall@2': -.1})),
    ('late_score_large', lambda r: r['per_query']['z'].update({'recall@2': 2})),
    ('late_huge_int', lambda r: r['per_query']['z'].update({'recall@2': 10**400})),
    ('missing_nonzero', lambda r: r['per_query']['z'].update({'recall@2': .5})),
    ('late_mean', lambda r: r['means'].update({'recall@2': .5})),
    ('late_mean_missing', lambda r: r['means'].pop('recall@2')),
    ('mean_bool', lambda r: r['means'].update({'hit@1': True})),
    ('count', lambda r: r.update(query_count=2)),
    ('count_bool', lambda r: r.update(query_count=True)),
    ('late_id', lambda r: r['manifest']['dataset']['query_ids'].__setitem__(-1, 'bad id')),
    ('unsorted_ids', lambda r: r['manifest']['dataset']['query_ids'].reverse()),
    ('duplicate_ids', lambda r: r['manifest']['dataset']['query_ids'].append('z')),
    ('missing_ids', lambda r: r.update(missing_queries=['unknown'])),
    ('ignored_ids', lambda r: r.update(ignored_queries=['z'])),
    ('gain', lambda r: r['manifest']['parameters'].update(gain='exponential')),
    ('metrics', lambda r: r['manifest']['parameters']['metrics'].reverse()),
    ('policy', lambda r: r['manifest']['parameters'].update(query_policy='intersection')),
    ('dataset_digest', lambda r: r['manifest']['dataset'].update(sha256='invalid')),
    ('run_digest', lambda r: r['manifest']['run'].update(sha256='A'*64)),
    ('checksum', lambda r: r['manifest'].update(artifact_sha256='0'*64)),
    # Valid digest syntax and consistent score payload still require checksum recomputation.
    ('valid_run_digest_edit', lambda r: r['manifest']['run'].update(sha256='0'*64)),
    ('valid_dataset_digest_edit', lambda r: r['manifest']['dataset'].update(sha256='0'*64)),
]


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('name,mutate', MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_corruption_is_not_dropped_or_trusted(archived, existing, name, mutate, monkeypatch, capsys):
    value = json.loads(archived.read_text())
    mutate(value)
    archived.write_text(json.dumps(value))
    output = archived.with_name('protected.json')
    if existing:
        output.write_bytes(b'sentinel\n')
    standalone(monkeypatch)
    assert main(args(archived, '--output', output, '--force')) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'error:' in captured.err
    if name in ('valid_run_digest_edit', 'valid_dataset_digest_edit', 'checksum'):
        assert 'checksum mismatch' in captured.err
    assert output.read_bytes() == b'sentinel\n' if existing else not output.exists()
    assert not list(output.parent.glob('.retgate-*'))


def test_reuses_reader_validator_and_recomputes_artifact_hash(archived, monkeypatch, capsys):
    import ragwrench.evalgate.cli as cli
    import ragwrench.evalgate.saved as saved
    calls = []
    reader, validator, checksum = cli.read_evaluation, saved.validate_evaluation, saved._artifact_sha256
    def read(path):
        calls.append('read')
        return reader(path)
    def validate(value):
        calls.append('validate')
        return validator(value)
    def digest(value):
        calls.append('checksum')
        return checksum(value)
    monkeypatch.setattr(cli, 'read_evaluation', read)
    monkeypatch.setattr(saved, 'validate_evaluation', validate)
    monkeypatch.setattr(saved, '_artifact_sha256', digest)
    standalone(monkeypatch)
    assert main(args(archived)) == 0
    assert calls == ['read', 'validate', 'checksum']
    assert capsys.readouterr().out == RECEIPT


def test_recomputed_consistent_edit_has_no_provenance_claim(archived, monkeypatch, capsys):
    from ragwrench.evalgate.saved import _artifact_sha256
    value = json.loads(archived.read_text())
    value['manifest']['run']['sha256'] = '0'*64
    value['manifest']['artifact_sha256'] = _artifact_sha256(value)
    archived.write_text(json.dumps(value))
    standalone(monkeypatch)
    assert main(args(archived)) == 0
    assert capsys.readouterr().out == RECEIPT


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
