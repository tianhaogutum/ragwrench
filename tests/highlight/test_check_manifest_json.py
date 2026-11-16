"""Machine-readable checks expose counts only after independent PDF and artifact checks."""
import json
from pathlib import Path

import pytest

from ragwrench.highlight.cli import main
from .test_check_manifest import manifest_case, reviewed_case


def result(capsys, args, code=0, category=None, path=None, message=None, counts=(3, 2)):
    directory = Path(args[1]).parent
    files = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    entries = sorted(directory.rglob('*'))
    capsys.readouterr()
    assert main(args + ['--json']) == code
    captured = capsys.readouterr()
    assert captured.err == ''
    assert captured.out.count('\n') == 1
    data = json.loads(captured.out)
    assert set(data) == {'format', 'schema_version', 'ok', 'summary', 'error'}
    assert data['format'] == 'citehl-check-manifest-result'
    assert type(data['schema_version']) is int and data['schema_version'] == 1
    assert type(data['ok']) is bool and data['ok'] is (code == 0)
    if code:
        assert data['summary'] is None
        assert data['error'] == {'category': category, 'path': path, 'message': data['error']['message']}
        assert isinstance(data['error']['message'], str)
        assert message in data['error']['message']
        assert 'Traceback' not in data['error']['message']
    else:
        assert data['error'] is None
        assert data['summary'] == dict(zip(('annotations', 'chunks'), counts))
        assert all(type(v) is int and v >= 0 for v in data['summary'].values())
    assert sorted(directory.rglob('*')) == entries
    assert {p: p.read_bytes() for p in files} == files
    return data


def test_check_manifest_json_success(manifest_case, capsys):
    result(capsys, manifest_case + ['--text-check', 'required'])


@pytest.mark.parametrize('keys,value,path', [
    (['schema_version'], True, "manifest['schema_version']"),
    (['pdf'], 'renamed.pdf', "manifest['pdf']"),
    (['chunks', 'summary/中文', 'pages'], [2, 1], "manifest['chunks']['summary/中文']['pages'][0]"),
    (['chunks', 'summary/中文', 'rects', 0, 'page'], 2.0,
     "manifest['chunks']['summary/中文']['rects'][0]['page']"),
    (['chunks', 'summary/中文', 'rects', 0, 'rect', 0], 72.005,
     "manifest['chunks']['summary/中文']['rects'][0]['rect'][0]"),
    (['chunks', 'summary/中文', 'rects', 0, 'source_ref'], 'forged',
     "manifest['chunks']['summary/中文']['rects'][0]['source_ref']"),
    (['chunks', 'summary/中文', 'rects', 1, 'source_refs'], ['merged/中文', '#/texts/0'],
     "manifest['chunks']['summary/中文']['rects'][1]['source_refs'][0]"),
    (['chunks', 'summary/中文', 'rects', 0, 'annotation_id'], 'forged',
     "manifest['chunks']['summary/中文']['rects'][0]['annotation_id']"),
    (['chunks', 'summary/中文', 'links'], ['raw name.pdf#page=1', 'raw name.pdf#page=2'],
     "manifest['chunks']['summary/中文']['links'][0]"),
    (['chunks', 'summary/中文', 'rects'], [], "manifest['chunks']['summary/中文']['rects']"),
    (['chunks', 'summary/中文'], {}, "manifest['chunks']['summary/中文']"),
    (['chunks'], {}, "manifest['chunks']"),
])
def test_json_manifest_mismatch_paths(manifest_case, capsys, keys, value, path):
    artifact = Path(manifest_case[4])
    data = json.loads(artifact.read_text())
    node = data
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    artifact.write_text(json.dumps(data))
    failed = result(capsys, manifest_case, 1, 'manifest_mismatch', path, 'regenerate with ragwrench highlight manifest')
    # Default diagnostics retain exactly the same human message and streams.
    assert main(manifest_case) == 1
    captured = capsys.readouterr()
    assert captured.out == '' and captured.err == 'ragwrench highlight: ' + failed['error']['message'] + '\n'


@pytest.mark.parametrize('target', ['pdf', 'regions', 'manifest'])
@pytest.mark.parametrize('change', ['missing', 'malformed', 'oversized'])
def test_json_input_failures(manifest_case, capsys, monkeypatch, target, change):
    path = Path(manifest_case[{'pdf': 1, 'regions': 2, 'manifest': 4}[target]])
    if change == 'missing':
        path.unlink()
    elif change == 'malformed':
        path.write_bytes(b'\xffbroken')
    else:
        monkeypatch.setattr('ragwrench.highlight.core.MAX_PDF_BYTES' if target == 'pdf' else 'ragwrench.highlight.core.MAX_JSON_BYTES', 10)
    result(capsys, manifest_case, 1, 'input_error', message='')


@pytest.mark.parametrize('payload', [
    '{"schema_version":0,"schema_versio\\u006e":1}',
    '{"chunks":{"same":{},"same":{}}}',
    '{"chunks":{"x":{"rects":[{"page":2,"page":1}]}}}',
    '{"chunks":NaN}', '{"chunks":Infinity}',
])
def test_json_duplicate_keys_and_constants(manifest_case, capsys, payload):
    Path(manifest_case[4]).write_text(payload)
    result(capsys, manifest_case, 1, 'input_error', message='invalid JSON')


@pytest.mark.parametrize('change,message', [
    ('bounds', 'visible page'), ('page', 'outside PDF'), ('duplicate', 'duplicate'),
])
def test_json_region_validation(manifest_case, capsys, change, message):
    from dataclasses import replace
    from ragwrench.highlight.core import read_regions, regions_jsonl
    path = Path(manifest_case[2])
    regions = read_regions(path)
    if change == 'bounds': regions[0] = replace(regions[0], rect=(0, 0, 10000, 10000))
    elif change == 'page': regions[0] = replace(regions[0], page=99)
    if change == 'duplicate':
        path.write_text(path.read_text() + json.dumps(regions[0].to_dict()) + '\n')
    else:
        path.write_text(regions_jsonl(regions))
    result(capsys, manifest_case, 1, 'input_error', message=message)


@pytest.mark.parametrize('change,message', [
    ('missing', 'missing annotation'), ('hidden', 'hidden/invisible'),
    ('duplicate', 'duplicate annotation'), ('rect', '/Rect mismatch'),
    ('metadata', 'invalid citehl region metadata'),
])
def test_json_pdf_verification(manifest_case, capsys, change, message):
    from pypdf import PdfWriter
    from pypdf.generic import ArrayObject, FloatObject, NameObject, NumberObject, TextStringObject
    from ragwrench.highlight.core import open_pdf
    path = Path(manifest_case[1])
    writer = PdfWriter(clone_from=open_pdf(path))
    page = writer.pages[0]
    if change == 'missing': page[NameObject('/Annots')] = ArrayObject()
    elif change == 'duplicate': page['/Annots'].append(page['/Annots'][0])
    else:
        key, value = {'hidden': ('/F', NumberObject(2)),
                      'rect': ('/Rect', ArrayObject([FloatObject(v) for v in (72, 686, 299, 700)])),
                      'metadata': ('/CiteHL', TextStringObject('{}'))}[change]
        page['/Annots'][0].get_object()[NameObject(key)] = value
    writer.write(path)
    # A second, invalid manifest cannot cause counts or hide the earlier PDF failure.
    Path(manifest_case[4]).write_text('{}')
    result(capsys, manifest_case, 1, 'verification_error', message=message)


@pytest.mark.parametrize('change,category,message', [
    ('unresolved', 'replay_error', 'unresolved'),
    ('stale', 'replay_error', 'stale or modified'),
    ('version', 'replay_error', 'stale or modified'),
    ('decision', 'replay_error', 'action'),
    ('missing', 'input_error', 'regular file'),
    ('malformed', 'input_error', 'invalid JSON'),
    ('duplicate', 'input_error', 'duplicate JSON key'),
])
def test_json_review_errors(reviewed_case, capsys, change, category, message):
    path = Path(reviewed_case[6])
    data = json.loads(path.read_text())
    if change == 'unresolved':
        next(i for i in data['items'] if i['region']['source_ref'] == 'blank')['resolution'] = None
    elif change == 'stale': data['document']['sha256'] = 'stale'
    elif change == 'version': data['schema_version'] = True
    elif change == 'decision': data['items'][0]['resolution'] = {'action': 'bogus', 'reason': 'bad'}
    if change == 'missing': path.unlink()
    elif change == 'malformed': path.write_text('{')
    elif change == 'duplicate': path.write_text('{"items":[],"items":[]}')
    else: path.write_text(json.dumps(data))
    result(capsys, reviewed_case, 1, category, message=message)


@pytest.mark.parametrize('mode,code', [('off', 0), ('best-effort', 0), ('required', 1)])
def test_json_reviewed_modes_and_manifest_loading_stage(reviewed_case, capsys, mode, code):
    result(capsys, reviewed_case + ['--text-check', mode], code,
           'verification_error', message='text unavailable', counts=(6, 6))
    # Loading manifest follows replay; a successful replay must reset the stage.
    Path(reviewed_case[4]).write_text('{')
    result(capsys, reviewed_case, 1, 'input_error', message='invalid JSON')


def test_json_explicit_empty_and_all_skip(reviewed_case, tmp_path, capsys, monkeypatch):
    review = Path(reviewed_case[6])
    data = json.loads(review.read_text())
    for item in data['items']:
        item['resolution'] = {'action': 'skip', 'reason': 'Reviewed omission'}
    review.write_text(json.dumps(data))
    cleared = tmp_path / 'cleared.pdf'
    assert main(['annotate', reviewed_case[1], reviewed_case[2], '--resolutions', str(review),
                 '-o', str(cleared), '--force']) == 0
    args = reviewed_case.copy()
    args[1] = str(cleared)
    empty = tmp_path / 'empty.jsonl'
    empty.write_text('\n \t\n')
    def forbidden(*args, **kwargs):
        pytest.fail('checker attempted a write')
    monkeypatch.setattr('ragwrench.highlight.core.atomic_write', forbidden)
    monkeypatch.setattr('ragwrench.highlight.pdf.atomic_write', forbidden)
    monkeypatch.setattr('ragwrench.highlight.cli.write_text', forbidden)
    result(capsys, args, counts=(0, 0))
    args[2] = str(empty)
    result(capsys, args + ['--allow-empty'], 1, 'input_error', message='expected 1..')
    args = args[:5]
    result(capsys, args, 1, 'input_error', message='expected 1..')
    result(capsys, args + ['--allow-empty', '--text-check', 'required'], counts=(0, 0))
    args[1] = reviewed_case[1]
    result(capsys, args + ['--allow-empty'], 1, 'verification_error', message='unexpected citehl annotation')
    args[1] = str(cleared)
    Path(args[4]).write_text('{"schema_version":1,"coordinate_space":"unrotated-pdf-user-space",'
                             '"pdf":"cleared.pdf","chunks":{"extra":{}}}')
    result(capsys, args + ['--allow-empty'], 1, 'manifest_mismatch', "manifest['chunks']", 'object fields differ')


def test_json_location_and_first_stage(manifest_case, tmp_path, capsys):
    args = manifest_case.copy()
    nested = tmp_path / 'nested'
    nested.mkdir()
    args[4] = str(nested / 'absent.json')
    result(capsys, args, 1, 'input_error', message='share a directory')
    Path(manifest_case[1]).unlink()
    Path(manifest_case[4]).write_text('{}')
    result(capsys, manifest_case, 1, 'input_error', message='regular file')


def test_json_tolerance_and_equivalent_manifest(manifest_case, capsys):
    from pypdf import PdfWriter
    from pypdf.generic import FloatObject
    from ragwrench.highlight.core import open_pdf
    artifact, pdf = Path(manifest_case[4]), Path(manifest_case[1])
    data = json.loads(artifact.read_text())
    for chunk in data['chunks'].values():
        for rect in chunk['rects']:
            rect['rect'] = [int(v) for v in rect['rect']]
    artifact.write_text(json.dumps({k: data[k] for k in reversed(data)}, indent=4))
    writer = PdfWriter(clone_from=open_pdf(pdf))
    rect = writer.pages[0]['/Annots'][0].get_object()['/Rect']
    rect[0] = FloatObject(float(rect[0]) + .005)
    writer.write(pdf)
    result(capsys, manifest_case)
    result(capsys, manifest_case + ['--tolerance', '.001'], 1, 'verification_error', message='/Rect mismatch')
    data['chunks']['other']['rects'][0]['rect'][0] += .005
    artifact.write_text(json.dumps(data))
    result(capsys, manifest_case + ['--tolerance', '100'], 1, 'manifest_mismatch',
           "manifest['chunks']['other']['rects'][0]['rect'][0]", 'numeric value differs')


def test_json_path_does_not_parse_message(manifest_case, capsys, monkeypatch):
    from ragwrench.highlight.manifest_check import ManifestMismatch
    monkeypatch.setattr(ManifestMismatch, '__str__', lambda self: 'Changed diagnostic without any path')
    Path(manifest_case[4]).write_text('[]')
    result(capsys, manifest_case, 1, 'manifest_mismatch', 'manifest', 'Changed diagnostic')


@pytest.mark.parametrize('arguments', [
    ['check-manifest'], ['check-manifest', 'a'], ['check-manifest', 'a', 'b'],
    ['check-manifest', 'a', 'b', '--manifest'],
    ['check-manifest', 'a', 'b', '--manifest', 'c', '--resolutions'],
    ['check-manifest', 'a', 'b', '--manifest', 'c', '-o', 'never.json'],
    ['check-manifest', 'a', 'b', '--manifest', 'c', '--force'],
    ['check-manifest', 'a', 'b', '--manifest', 'c', '--unknown'],
    ['check-manifest', 'a', 'b', '--manifest', 'c', '--tolerance', 'nan'],
    ['check-manifest', 'a', 'b', '--manifest', 'c', '--tolerance', 'inf'],
    ['check-manifest', 'a', 'b', '--manifest', 'c', '--tolerance', '-1'],
    ['check-manifest', 'a', 'b', '--manifest', 'c', '--text-check', 'guess'],
    ['manifest', 'a', 'b', '-o', 'never.json'],
])
def test_json_usage_errors(capsys, arguments):
    with pytest.raises(SystemExit) as exc:
        main(arguments + ['--json'])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'error:' in captured.err
