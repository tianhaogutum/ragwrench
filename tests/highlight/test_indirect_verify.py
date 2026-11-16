"""Persisted PDF text-string identity is independent of direct/indirect storage."""
import json
from dataclasses import replace

import pytest
from pypdf import PdfWriter
from pypdf.generic import (DictionaryObject, IndirectObject, NameObject, NumberObject,
                           RectangleObject, TextStringObject)

from ragwrench.highlight.cli import main
from ragwrench.highlight.core import CiteError, canonical, open_pdf, read_regions
from ragwrench.highlight.pdf import (PREFIX, annotate, annotation_id, embedded_region, manifest,
                        region_metadata, verify)
from ragwrench.highlight.sidecar import export_sidecar


def indirect_names(writer, *, legacy=False):
    for page in writer.pages:
        for ref in page.get('/Annots', []):
            annotation = ref.get_object()
            annotation[NameObject('/NM')] = writer._add_object(annotation['/NM'])
            if legacy:
                annotation.pop('/CiteHL', None)


@pytest.fixture(params=[False, True], ids=['embedded', 'legacy'])
def indirect_case(request, examples, tmp_path):
    regions = examples / 'regions.jsonl'
    pdf = tmp_path / 'indirect.pdf'
    annotate(open_pdf(examples / 'sample.pdf'), read_regions(regions), pdf)
    writer = PdfWriter(clone_from=open_pdf(pdf))
    indirect_names(writer, legacy=request.param)
    writer.write(pdf)
    return pdf, regions, request.param


def test_verify_indirect_persisted_ids(indirect_case, capsys):
    pdf, regions, _ = indirect_case
    before = {p: p.read_bytes() for p in (pdf, regions)}
    entries = sorted(pdf.parent.iterdir())
    reader = open_pdf(pdf)
    names = [p['/Annots'][0].get_object().raw_get('/NM') for p in reader.pages]
    assert all(isinstance(n, IndirectObject) for n in names)
    expected = {'annotations': 2, 'text_checked': 2, 'text_skipped': 0,
                'text_check': 'required', 'notes': []}
    assert verify(reader, read_regions(regions), text_check='required') == expected
    assert all(p['/Annots'][0].get_object().raw_get('/NM') is n
               for p, n in zip(reader.pages, names))
    for structured in (False, True):
        capsys.readouterr()
        args = ['verify', str(pdf), str(regions), '--text-check', 'required']
        assert main(args + (['--json'] if structured else [])) == 0
        captured = capsys.readouterr()
        assert captured.err == ''
        if structured:
            assert json.loads(captured.out) == {
                'format': 'citehl-verify-result', 'schema_version': 1, 'ok': True,
                'summary': expected, 'error': None}
            assert captured.out.count('\n') == 1
        else:
            assert captured.out == 'OK 2/2 ' + json.dumps(expected) + '\n'
    assert {p: p.read_bytes() for p in before} == before
    assert sorted(pdf.parent.iterdir()) == entries


@pytest.mark.parametrize('change,message', [
    ('metadata-id', '/NM mismatch'), ('region-hash', '/NM mismatch'),
    ('metadata-key', 'duplicate JSON key'), ('metadata-version', 'schema_version'),
    ('renamed', '/NM mismatch'), ('missing-name', '/NM mismatch'),
    ('non-text', '/NM mismatch'), ('rect', '/Rect mismatch'),
    ('quad', '/QuadPoints mismatch'), ('contents', '/Contents mismatch'),
    ('subtype', 'wrong page or subtype'), ('hidden', 'hidden/invisible'),
    ('opacity', 'hidden/invisible'), ('moved', 'wrong page or subtype'),
    ('missing', 'missing annotation'), ('extra', 'unexpected citehl annotation'),
    ('duplicate-ref', 'duplicate annotation'), ('duplicate-direct', 'duplicate annotation'),
    ('duplicate-indirect', 'duplicate annotation'), ('duplicate-cross-page', 'duplicate annotation'),
])
def test_indirect_integrity(indirect_case, capsys, change, message):
    pdf, regions, legacy = indirect_case
    writer = PdfWriter(clone_from=open_pdf(pdf))
    page = writer.pages[0]
    ref = page['/Annots'][0]
    annotation = ref.get_object()
    # Metadata corruptions are tested as present payloads, never legacy fallbacks.
    if change in ('metadata-id', 'region-hash', 'metadata-key', 'metadata-version',
                  'renamed', 'missing-name'):
        annotation[NameObject('/CiteHL')] = TextStringObject(region_metadata(read_regions(regions)[0]))
    if change in ('metadata-id', 'region-hash', 'metadata-key', 'metadata-version'):
        data = json.loads(annotation['/CiteHL'])
        if change == 'metadata-id': data['annotation_id'] = annotation_id(read_regions(regions)[1])
        elif change == 'region-hash': data['region']['text'] = 'changed'
        elif change == 'metadata-version': data['schema_version'] = True
        raw = canonical(data)
        if change == 'metadata-key': raw = '{"schema_version":1,' + raw[1:]
        annotation[NameObject('/CiteHL')] = TextStringObject(raw)
    elif change == 'renamed':
        annotation[NameObject('/NM')] = writer._add_object(TextStringObject('foreign'))
    elif change == 'missing-name': annotation.pop('/NM')
    elif change == 'non-text':
        annotation[NameObject('/NM')] = writer._add_object(NumberObject(17))
        if legacy:
            message = 'missing annotation'
    elif change == 'missing': page['/Annots'].pop()
    elif change == 'extra':
        regions = pdf.with_suffix('.jsonl')
        # Supply a subset; the second, otherwise valid annotation is unexpected.
        from ragwrench.highlight.core import regions_jsonl
        regions.write_text(regions_jsonl(read_regions(indirect_case[1])[:1]))
    elif change == 'moved': writer.pages[1]['/Annots'].append(page['/Annots'].pop())
    elif change == 'duplicate-ref': page['/Annots'].append(ref)
    elif change.startswith('duplicate-'):
        duplicate = DictionaryObject(dict(annotation))
        if change == 'duplicate-direct': duplicate[NameObject('/NM')] = annotation['/NM']
        elif change == 'duplicate-indirect':
            duplicate[NameObject('/NM')] = writer._add_object(TextStringObject(annotation['/NM']))
        target = writer.pages[1] if change == 'duplicate-cross-page' else page
        target['/Annots'].append(writer._add_object(duplicate))
    else:
        key, value = {
            'rect': ('/Rect', RectangleObject((73, 686, 300, 700))),
            'quad': ('/QuadPoints', RectangleObject((1, 2, 3, 4))),
            'contents': ('/Contents', TextStringObject('edited')),
            'subtype': ('/Subtype', NameObject('/Square')),
            'hidden': ('/F', NumberObject(2)), 'opacity': ('/CA', NumberObject(0)),
        }[change]
        annotation[NameObject(key)] = value
    writer.write(pdf)
    before = {p: p.read_bytes() for p in (pdf, regions)}
    for structured in (False, True):
        capsys.readouterr()
        args = ['verify', str(pdf), str(regions), '--text-check', 'off']
        assert main(args + (['--json'] if structured else [])) == 1
        captured = capsys.readouterr()
        if structured:
            result = json.loads(captured.out)
            assert result['ok'] is False and result['summary'] is None
            assert result['error']['category'] == 'verification_error'
            assert message in result['error']['message']
            assert captured.err == '' and captured.out.count('\n') == 1
        else:
            assert captured.out == '' and message in captured.err
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('metadata', [False, True], ids=['legacy', 'embedded'])
def test_verify_rejects_name_typed_citehl_id(indirect_case, capsys, metadata):
    # A PDF name is not a text string, but a citehl-prefixed name must not be
    # silently treated as foreign and let verify succeed.
    pdf, regions, _ = indirect_case
    writer = PdfWriter(clone_from=open_pdf(pdf))
    extra = DictionaryObject({
        NameObject('/Subtype'): NameObject('/Text'),
        NameObject('/Rect'): RectangleObject((1, 1, 10, 10)),
        NameObject('/NM'): NameObject('/' + PREFIX + 'a' * 64),
    })
    if metadata:
        extra[NameObject('/CiteHL')] = TextStringObject(region_metadata(read_regions(regions)[0]))
    writer.add_annotation(0, extra)
    writer.write(pdf)
    before = pdf.read_bytes()
    assert main(['verify', str(pdf), str(regions), '--text-check', 'off', '--json']) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['ok'] is False and result['summary'] is None
    assert 'must be a PDF text string' in result['error']['message'] or '/NM mismatch' in result['error']['message']
    assert pdf.read_bytes() == before


def test_indirect_integer_json_identity(examples, tmp_path):
    region = replace(read_regions(examples / 'regions.jsonl')[0], rect=(72, 686, 300, 700),
                     source_refs=('#/texts/0', '另一个来源'))
    floated = replace(region, rect=tuple(map(float, region.rect)))
    assert annotation_id(region) != annotation_id(floated)
    pdf = tmp_path / 'integer.pdf'
    annotate(open_pdf(examples / 'sample.pdf'), [region], pdf)
    writer = PdfWriter(clone_from=open_pdf(pdf))
    indirect_names(writer)
    writer.write(pdf)
    before = pdf.read_bytes()
    reader = open_pdf(pdf)
    annotation = reader.pages[0]['/Annots'][0].get_object()
    raw = annotation.raw_get('/NM')
    assert isinstance(raw, IndirectObject)
    recovered = embedded_region(annotation)
    assert all(type(n) is int for n in recovered.rect)
    assert annotation_id(recovered) == annotation_id(region)
    assert verify(reader, [region], text_check='required')['text_checked'] == 1
    with pytest.raises(CiteError, match='missing annotation'):
        verify(reader, [floated], text_check='off')
    assert annotation.raw_get('/NM') is raw and pdf.read_bytes() == before


def test_indirect_readonly_consumers(indirect_case, capsys):
    pdf, regions, legacy = indirect_case
    saved_manifest, saved_sidecar = pdf.with_suffix('.manifest.json'), pdf.with_suffix('.sidecar.json')
    expected_regions = read_regions(regions)
    saved_manifest.write_text(canonical(manifest(expected_regions, pdf.name)))
    # Independently establish expected sidecar from direct-ID annotations.
    writer = PdfWriter(clone_from=open_pdf(pdf))
    for page in writer.pages:
        annotation = page['/Annots'][0].get_object()
        annotation[NameObject('/NM')] = annotation['/NM']
    direct = pdf.with_name('direct.pdf')
    writer.write(direct)
    expected = export_sidecar(open_pdf(direct), expected_regions)
    saved_sidecar.write_text(canonical(expected))
    before = {p: p.read_bytes() for p in (pdf, regions, saved_manifest, saved_sidecar)}
    for structured in (False, True):
        assert main(['check-manifest', str(pdf), str(regions), '--manifest', str(saved_manifest),
                     '--text-check', 'required'] + (['--json'] if structured else [])) == 0
    assert export_sidecar(open_pdf(pdf), expected_regions) == expected
    assert main(['check-sidecar', str(pdf), '--sidecar', str(saved_sidecar), '--regions', str(regions)]) == 0
    capsys.readouterr()
    if legacy:
        # Default PDF-only export has no provenance source for legacy IDs.
        with pytest.raises(CiteError): export_sidecar(open_pdf(pdf))
        assert main(['check-sidecar', str(pdf), '--sidecar', str(saved_sidecar)]) == 1
        assert capsys.readouterr().out == ''
    else:
        assert export_sidecar(open_pdf(pdf)) == expected
        assert main(['check-sidecar', str(pdf), '--sidecar', str(saved_sidecar)]) == 0
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('mode', ['required', 'best-effort', 'off'])
@pytest.mark.parametrize('evidence', ['mismatch', 'unavailable'])
def test_indirect_text_modes(examples, tmp_path, capsys, mode, evidence):
    from ragwrench.highlight.core import regions_jsonl
    region = read_regions(examples / 'regions.jsonl')[0]
    if evidence == 'mismatch': region = replace(region, text='elephants sunshine')
    else: region = replace(region, rect=(10.0, 10.0, 40.0, 40.0))
    pdf, regions = tmp_path / 'text.pdf', tmp_path / 'text.jsonl'
    regions.write_text(regions_jsonl([region]))
    annotate(open_pdf(examples / 'sample.pdf'), [region], pdf)
    writer = PdfWriter(clone_from=open_pdf(pdf))
    indirect_names(writer)
    writer.write(pdf)
    before = {p: p.read_bytes() for p in (pdf, regions)}
    code = int(mode != 'off' and (evidence == 'mismatch' or mode == 'required'))
    assert main(['verify', str(pdf), str(regions), '--text-check', mode, '--json']) == code
    result = json.loads(capsys.readouterr().out)
    if code:
        assert result['summary'] is None and evidence in result['error']['message']
    else:
        assert result['summary'] == {
            'annotations': 1, 'text_checked': 0,
            'text_skipped': int(mode == 'best-effort'), 'text_check': mode, 'notes': []}
    assert {p: p.read_bytes() for p in before} == before


def test_mixed_direct_indirect_and_foreign(indirect_case):
    pdf, regions, _ = indirect_case
    writer = PdfWriter(clone_from=open_pdf(pdf))
    second = writer.pages[1]['/Annots'][0].get_object()
    second[NameObject('/NM')] = second['/NM']
    writer.add_annotation(0, DictionaryObject({
        NameObject('/Subtype'): NameObject('/Text'),
        NameObject('/Rect'): RectangleObject((1, 1, 10, 10)),
        NameObject('/NM'): writer._add_object(TextStringObject('foreign')),
    }))
    writer.write(pdf)
    before = pdf.read_bytes()
    assert verify(open_pdf(pdf), read_regions(regions), text_check='required')['text_checked'] == 2
    assert pdf.read_bytes() == before
