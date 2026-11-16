"""Guard diagnostic contracts for --expected-preflight and --preflight-diff-output."""
import copy
import json
from collections import Counter

import pytest

from ragwrench.evalgate.cli import main
from ragwrench.evalgate.formats import InputError
from ragwrench.evalgate.preflight import diff_preflight
from .test_preflight import api, argv, family, forbid_inference


def setup(family, saved, *, drift=True):
    folder = family[-1][0].parent
    expected = api(family, not saved, preflight_only=True)
    if drift:
        expected['settings']['seed'] = 43
        expected['settings']['confidence'] = .9
    audit = folder / 'expected.json'
    audit.write_text(json.dumps(expected))
    return expected, audit, folder / 'drift.json', folder / 'normal.json'


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('preflight_only', [False, True])
def test_api_valid_drift_has_owned_standalone_diff(family, saved, preflight_only, monkeypatch):
    from ragwrench.evalgate.compare import PreflightMismatch
    expected, _, _, _ = setup(family, saved)
    current = api(family, saved, preflight_only=True)
    wanted = diff_preflight(expected, current)
    before = copy.deepcopy((family[:-1], expected))
    forbid_inference(monkeypatch)
    with pytest.raises(PreflightMismatch, match='expected preflight mismatch') as caught:
        api(family, saved, expected_preflight=expected, preflight_only=preflight_only)
    assert isinstance(caught.value, InputError)
    assert caught.value.diff == wanted
    assert [r['path'] for r in wanted['changes']] == ['/settings/confidence', '/settings/seed']
    caught.value.diff['changes'][0]['expected'] = 'edited'
    assert (family[:-1], expected) == before


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('preflight_only', [False, True])
def test_cli_drift_file_no_report_or_inference(family, saved, preflight_only, monkeypatch, capsys):
    expected, audit, diagnostic, output = setup(family, saved)
    current = api(family, saved, preflight_only=True)
    output.write_text('normal sentinel')
    forbid_inference(monkeypatch)
    extra = ['--preflight-only'] if preflight_only else []
    assert main(argv(family, saved, '--expected-preflight', audit, '--preflight-diff-output', diagnostic,
                     '--output', output, '--force', *extra)) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'expected preflight mismatch' in captured.err
    assert output.read_text() == 'normal sentinel'
    assert json.loads(diagnostic.read_text()) == diff_preflight(expected, current)
    assert list(diagnostic.parent.glob('.retgate-*')) == []


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('drop', [.49, .5])
@pytest.mark.parametrize('existing', [False, True])
def test_cli_match_retains_normal_reports_and_leaves_diagnostic(family, saved, drop, existing, capsys):
    expected, audit, diagnostic, output = setup(family, saved, drift=False)
    expected = api(family, not saved, thresholds={'hit@1': drop}, preflight_only=True)
    audit.write_text(json.dumps(expected))
    if existing:
        diagnostic.write_text('stale sentinel')
    normal = api(family, saved, thresholds={'hit@1': drop})
    args = argv(family, saved, '--expected-preflight', audit, '--preflight-diff-output', diagnostic,
                '--output', output, '--force', '--format', 'json')
    args[args.index('--max-drop') + 1] = f'hit@1={drop}'
    assert main(args) == (1 if drop == .49 else 0)
    assert capsys.readouterr().err == ''
    assert json.loads(output.read_text()) == normal
    assert (diagnostic.read_text() == 'stale sentinel' if existing else not diagnostic.exists())


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('kind', ['malformed', 'unbound', 'null', 'duplicate', 'nonfinite', 'version',
    'late_binding', 'bad_policy', 'late_candidate', 'bad_options', 'budget'])
def test_errors_have_no_diagnostic_and_preserve_both_outputs(family, saved, kind, monkeypatch, capsys):
    expected, audit, diagnostic, output = setup(family, saved)
    diagnostic.write_text('diff sentinel')
    output.write_text('normal sentinel')
    extra = []
    if kind in ('malformed', 'unbound', 'null', 'duplicate', 'nonfinite'):
        audit.write_text({'malformed': '{', 'unbound': '{}', 'null': 'null',
                          'duplicate': '{"schema_version":1,"schema_version":1}',
                          'nonfinite': '{"value":NaN}'}[kind])
    elif kind == 'version':
        expected['schema_version'] = 2
    elif kind == 'late_binding':
        expected['input_bindings']['candidates']['strict']['artifact_sha256'] = 'bad'
    elif kind == 'bad_policy':
        expected['policy']['effective_thresholds']['strict'] = {}
    elif kind == 'late_candidate':
        bad = audit.parent / 'bad'
        bad.write_text('{}' if saved else 'broken\n')
        extra = ['--candidate', f'z_invalid={bad}']
    elif kind == 'bad_options':
        extra = ['--samples', '99']
    else:
        extra = ['--max-family-pairs', '3']
    if kind in ('version', 'late_binding', 'bad_policy'):
        audit.write_text(json.dumps(expected))
    before = audit.read_bytes()
    forbid_inference(monkeypatch)
    assert main(argv(family, saved, '--expected-preflight', audit, '--preflight-diff-output', diagnostic,
                     '--output', output, '--force', *extra)) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'error:' in captured.err
    assert diagnostic.read_text() == 'diff sentinel'
    assert output.read_text() == 'normal sentinel'
    assert audit.read_bytes() == before
    if kind in ('unbound', 'null', 'version', 'late_binding', 'bad_policy') and expected is not None:
        bad = json.loads(audit.read_text())
        if bad is not None:
            with pytest.raises(InputError) as caught:
                api(family, saved, expected_preflight=bad)
            assert not hasattr(caught.value, 'diff')


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('matching', [False, True])
@pytest.mark.parametrize('kind', ['requires_guard', 'existing', 'directory', 'missing_parent', 'symlink',
    'normal_equal', 'normal_symlink', 'normal_hardlink', 'normal_input', 'normal_existing', 'normal_missing_parent'])
def test_invalid_destinations_reject_before_inference(family, saved, matching, kind, monkeypatch, capsys):
    _, audit, diagnostic, output = setup(family, saved, drift=not matching)
    output.write_text('normal sentinel')
    force = ['--force']
    guard = ['--expected-preflight', audit]
    if kind == 'requires_guard':
        guard = []
    elif kind == 'existing':
        diagnostic.write_text('diff sentinel')
        force = []
        output.unlink()
    elif kind == 'directory':
        diagnostic.mkdir()
    elif kind == 'missing_parent':
        diagnostic = diagnostic / 'missing'
    elif kind == 'symlink':
        diagnostic.symlink_to(audit.parent / 'missing-target')
    elif kind == 'normal_equal':
        diagnostic = output
    elif kind == 'normal_symlink':
        diagnostic.symlink_to(output)
    elif kind == 'normal_hardlink':
        diagnostic.hardlink_to(output)
    elif kind == 'normal_input':
        output = family[-1][4 if saved else 0]
    elif kind == 'normal_existing':
        force = []
    else:
        output = output / 'missing'
    before = {p: p.read_bytes() for p in [*family[-1], audit, output, diagnostic] if p.is_file()}
    forbid_inference(monkeypatch)
    assert main(argv(family, saved, *guard, '--preflight-diff-output', diagnostic, '--output', output, *force)) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'error:' in captured.err
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not list(audit.parent.glob('.retgate-*'))


@pytest.mark.parametrize('saved,target', [(False, t) for t in ('expected', 'baseline', 'candidate', 'qrels')] +
                         [(True, t) for t in ('expected', 'baseline', 'candidate')])
@pytest.mark.parametrize('alias', ['same', 'relative', 'symlink', 'hardlink'])
def test_diagnostic_cannot_alias_any_input(family, saved, target, alias, monkeypatch, capsys):
    _, audit, diagnostic, output = setup(family, saved)
    files = family[-1]
    source = {'expected': audit, 'baseline': files[3 if saved else 1],
              'candidate': files[4 if saved else 2], 'qrels': files[0]}[target]
    if alias == 'same':
        diagnostic = source
    elif alias == 'relative':
        diagnostic = source.parent / '..' / source.parent.name / source.name
    elif alias == 'symlink':
        diagnostic.symlink_to(source)
    else:
        diagnostic.hardlink_to(source)
    before = source.read_bytes()
    forbid_inference(monkeypatch)
    assert main(argv(family, saved, '--expected-preflight', audit, '--preflight-diff-output', diagnostic,
                     '--output', output, '--force')) == 2
    assert capsys.readouterr().out == ''
    assert source.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('stage', ['write', 'fsync', 'replace', 'link'])
def test_atomic_failure_preserves_outputs(family, saved, stage, monkeypatch, capsys):
    import ragwrench.evalgate.cli as cli
    _, audit, diagnostic, output = setup(family, saved)
    output.write_text('normal sentinel')
    force = ['--force']
    if stage == 'link':
        force = []
        output.unlink()
    else:
        diagnostic.write_text('diff sentinel')
    def fail(*args, **kwargs):
        raise OSError('synthetic diagnostic write failure')
    if stage == 'write':
        original = cli.tempfile.NamedTemporaryFile
        class BrokenStream:
            def __init__(self, **kwargs):
                self.stream = original(**kwargs)
                self.name = self.stream.name
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            write = fail
        monkeypatch.setattr(cli.tempfile, 'NamedTemporaryFile', BrokenStream)
    else:
        monkeypatch.setattr(cli.os, stage, fail)
    forbid_inference(monkeypatch)
    assert main(argv(family, saved, '--expected-preflight', audit, '--preflight-diff-output', diagnostic,
                     '--output', output, *force)) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'synthetic diagnostic write failure' in captured.err
    assert (diagnostic.read_text() == 'diff sentinel' if stage != 'link' else not diagnostic.exists())
    assert (output.read_text() == 'normal sentinel' if stage != 'link' else not output.exists())
    assert not list(audit.parent.glob('.retgate-*'))


@pytest.mark.parametrize('saved', [False, True])
def test_drift_reuses_reads_and_scores_with_no_original_saved_inputs(family, saved, monkeypatch, capsys):
    import ragwrench.evalgate.cli as cli
    import ragwrench.evalgate.compare as comparison
    expected, audit, diagnostic, _ = setup(family, saved)
    calls, scores = [], []
    for name in ('read_qrels', 'read_run', 'read_evaluation'):
        original = getattr(cli, name)
        def count(path, *args, _original=original, **kwargs):
            calls.append(path)
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(cli, name, count)
    evaluate = comparison.evaluate
    def score(*args, **kwargs):
        scores.append(1)
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(comparison, 'evaluate', score)
    def forbidden(*args, **kwargs):
        pytest.fail('diagnostic reread original inputs or rescored saved artifacts')
    monkeypatch.setattr('ragwrench.evalgate.saved.evaluate', forbidden)
    monkeypatch.setattr('ragwrench.evalgate.formats.read_qrels', forbidden)
    monkeypatch.setattr('ragwrench.evalgate.formats.read_run', forbidden)
    if saved:
        for path in family[-1][:3]:
            path.unlink()
    forbid_inference(monkeypatch)
    assert main(argv(family, saved, '--expected-preflight', audit, '--preflight-diff-output', diagnostic)) == 2
    assert capsys.readouterr().out == ''
    q, b, c, bs, cs = family[-1]
    assert Counter(calls) == (Counter({bs: 1, cs: 2}) if saved else Counter({q: 1, b: 1, c: 2}))
    assert len(scores) == (0 if saved else 3)
    assert [r['path'] for r in json.loads(diagnostic.read_text())['changes']] == ['/settings/confidence', '/settings/seed']


@pytest.mark.parametrize('saved', [False, True])
def test_identity_policy_and_settings_all_paths_without_inference(family, saved, monkeypatch):
    from ragwrench.evalgate.compare import PreflightMismatch
    from ragwrench.evalgate.saved import saved_evaluation
    expected = api(family, not saved, preflight_only=True)
    changed = copy.deepcopy(family)
    q, b, c, m, artifacts, _ = changed
    c['q2']['d'] += 10
    artifacts[1] = saved_evaluation(q, c, m)
    options = dict(thresholds={'hit@1': .5}, seed=44, confidence=.9, max_family_pairs=5)
    current = api(changed, saved, preflight_only=True, **options)
    wanted = diff_preflight(expected, current)
    paths = [r['path'] for r in wanted['changes']]
    assert paths == sorted(paths) and len(paths) >= 10
    assert '/input_bindings/candidates/strict/run_sha256' in paths
    assert '/policy/effective_thresholds/strict/hit@1' in paths
    assert '/settings/seed' in paths
    forbid_inference(monkeypatch)
    with pytest.raises(PreflightMismatch) as caught:
        api(changed, saved, expected_preflight=expected, **options)
    assert caught.value.diff == wanted


@pytest.mark.parametrize('saved', [False, True])
def test_force_replaces_only_drift_diagnostic(family, saved, monkeypatch, capsys):
    expected, audit, diagnostic, output = setup(family, saved)
    current = api(family, saved, preflight_only=True)
    diagnostic.write_text('old diagnostic')
    before = audit.read_bytes()
    forbid_inference(monkeypatch)
    assert main(argv(family, saved, '--expected-preflight', audit, '--preflight-diff-output', diagnostic,
                     '--output', output, '--force')) == 2
    assert capsys.readouterr().out == ''
    assert json.loads(diagnostic.read_text()) == diff_preflight(expected, current)
    assert audit.read_bytes() == before and not output.exists()


@pytest.mark.parametrize('saved', [False, True])
def test_matching_preflight_emits_audit_only(family, saved, monkeypatch, capsys):
    _, audit, diagnostic, _ = setup(family, saved, drift=False)
    current = api(family, saved, preflight_only=True)
    forbid_inference(monkeypatch)
    assert main(argv(family, saved, '--expected-preflight', audit, '--preflight-diff-output', diagnostic,
                     '--preflight-only')) == 0
    captured = capsys.readouterr()
    assert captured.err == '' and json.loads(captured.out) == current
    assert not diagnostic.exists()


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('kind', ['malformed', 'unbound', 'late_binding'])
def test_invalid_audits_create_neither_output(family, saved, kind, monkeypatch, capsys):
    expected, audit, diagnostic, output = setup(family, saved)
    if kind == 'late_binding':
        expected['input_bindings']['candidates']['strict']['run_sha256'] = 'invalid'
        audit.write_text(json.dumps(expected))
    else:
        audit.write_text('{' if kind == 'malformed' else '{}')
    forbid_inference(monkeypatch)
    assert main(argv(family, saved, '--expected-preflight', audit, '--preflight-diff-output', diagnostic,
                     '--output', output, '--force')) == 2
    assert capsys.readouterr().out == ''
    assert not diagnostic.exists() and not output.exists()
