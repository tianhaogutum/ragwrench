"""Read-only v1 inventory: exact /NM identity, current native geometry, provenance.

Root fields: format, schema_version, coordinate_space, annotations, summary.
Entries: annotation_id (exact string), page (1-based int), rect (four finite
ordered numbers), metadata_available (bool), region (full normalized object/null).
Page/array order, no foreign entries; summary has owned/foreign integer counts.
No files or partial stdout on error. Legacy means absent /CiteHL, not invalid.
"""
import json
from dataclasses import replace

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, RectangleObject, TextStringObject

from ragwrench.highlight.cli import main
from ragwrench.highlight.core import open_pdf, read_regions
from ragwrench.highlight.pdf import add_highlight, annotation_id
from ragwrench.highlight.sidecar import COORDINATE_SPACE


@pytest.fixture
def inventory_pdf(examples, tmp_path):
    regions = read_regions(examples / 'regions.jsonl')
    regions[0] = replace(regions[0], source_ref='原文', source_refs=('merged', '原文'))
    writer = PdfWriter(clone_from=open_pdf(examples / 'sample.pdf'))
    add_highlight(writer, regions[0])
    writer.add_annotation(0, DictionaryObject({
        NameObject('/Subtype'): NameObject('/Highlight'),
        NameObject('/T'): TextStringObject('citehl'),
        NameObject('/Contents'): TextStringObject(annotation_id(regions[0])),
        NameObject('/Rect'): RectangleObject((1, 2, 3, 4)),
    }))
    legacy = replace(regions[0], chunk_id='legacy')
    add_highlight(writer, legacy)
    annotation = writer.pages[0]['/Annots'][-1].get_object()
    annotation.pop('/CiteHL')
    annotation[NameObject('/NM')] = TextStringObject('citehl:v1:' + '0' * 64)
    add_highlight(writer, regions[1])
    pdf = tmp_path / 'inventory.pdf'
    writer.write(pdf)
    return pdf, regions, legacy


def test_inventory_exact_order_metadata_and_read_only(inventory_pdf, tmp_path, capsys):
    pdf, regions, legacy = inventory_pdf
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert main(['list-annotations', str(pdf)]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    assert captured.out.endswith('\n') and len(captured.out.splitlines()) == 1
    assert json.loads(captured.out) == {
        'format': 'citehl-annotation-inventory', 'schema_version': 1,
        'coordinate_space': COORDINATE_SPACE,
        'annotations': [
            {'annotation_id': annotation_id(regions[0]), 'page': 1,
             'rect': list(regions[0].rect), 'metadata_available': True, 'region': regions[0].to_dict()},
            {'annotation_id': 'citehl:v1:' + '0' * 64, 'page': 1,
             'rect': list(legacy.rect), 'metadata_available': False, 'region': None},
            {'annotation_id': annotation_id(regions[1]), 'page': 2,
             'rect': list(regions[1].rect), 'metadata_available': True, 'region': regions[1].to_dict()},
        ],
        'summary': {'owned': 3, 'foreign': 1},
    }
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_current_geometry_and_page_do_not_regenerate_identity(inventory_pdf, capsys, rotation):
    pdf, regions, legacy = inventory_pdf
    writer = PdfWriter(clone_from=open_pdf(pdf))
    writer.pages[0].rotate(rotation)
    first = writer.pages[0]['/Annots'][0].get_object()
    first[NameObject('/Rect')] = RectangleObject((-20, -10, 21, 31))
    first[NameObject('/Contents')] = TextStringObject('viewer edited')
    first[NameObject('/T')] = TextStringObject('viewer')
    first[NameObject('/Subtype')] = NameObject('/Square')
    # Move the metadata-bearing page-2 entry onto page 1 without rewriting it.
    writer.pages[0]['/Annots'].append(writer.pages[1]['/Annots'].pop())
    writer.write(pdf)
    before = pdf.read_bytes()
    assert main(['list-annotations', str(pdf)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['annotations'][0] == {
        'annotation_id': annotation_id(regions[0]), 'page': 1,
        'rect': [-20, -10, 21, 31], 'metadata_available': True, 'region': regions[0].to_dict(),
    }
    moved = result['annotations'][2]
    assert moved['page'] == 1 and moved['region']['page'] == 2
    assert moved['annotation_id'] == annotation_id(regions[1])
    assert pdf.read_bytes() == before


def test_indirect_arrays_ids_rects_and_metadata(inventory_pdf, capsys):
    pdf, regions, _ = inventory_pdf
    writer = PdfWriter(clone_from=open_pdf(pdf))
    first = writer.pages[0]['/Annots'][0].get_object()
    for key in ['/NM', '/Rect', '/CiteHL']:
        first[NameObject(key)] = writer._add_object(first[key])
    for page in writer.pages:
        page[NameObject('/Annots')] = writer._add_object(page['/Annots'])
    writer.write(pdf)
    assert main(['list-annotations', str(pdf)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['annotations'][0]['annotation_id'] == annotation_id(regions[0])
    assert result['annotations'][0]['region'] == regions[0].to_dict()
    assert result['summary'] == {'owned': 3, 'foreign': 1}


@pytest.mark.parametrize('foreign', [0, 1, 3])
def test_empty_owned_set(examples, tmp_path, capsys, foreign):
    writer = PdfWriter(clone_from=open_pdf(examples / 'sample.pdf'))
    for i in range(foreign):
        # Duplicate foreign IDs, appearance/title/text, invalid foreign rects
        # and non-v1 names are not an ownership heuristic.
        writer.add_annotation(i % 2, DictionaryObject({
            NameObject('/Subtype'): NameObject('/Highlight'),
            NameObject('/T'): TextStringObject('citehl'),
            NameObject('/Contents'): TextStringObject('citehl:v1:' + '1' * 64),
            NameObject('/NM'): TextStringObject('ragwrench highlight:v2:foreign'),
        }))
    pdf = tmp_path / 'empty.pdf'
    writer.write(pdf)
    before = pdf.read_bytes()
    assert main(['list-annotations', str(pdf)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['annotations'] == [] and data['summary'] == {'owned': 0, 'foreign': foreign}
    assert pdf.read_bytes() == before


@pytest.mark.parametrize('change', [
    'duplicate-ref', 'duplicate-dictionary', 'duplicate-legacy',
    'short-id', 'uppercase-hex', 'trailing-space',
    'metadata-json', 'metadata-type', 'metadata-version', 'metadata-id',
    'metadata-region', 'metadata-extra', 'metadata-duplicate-key', 'renamed', 'no-nm',
    'missing-rect', 'short-rect', 'string-rect', 'boolean-rect', 'inverted-rect',
    'null-annots', 'dict-annots', 'nondict-entry',
])
def test_failure_is_complete_and_read_only(inventory_pdf, tmp_path, capsys, change):
    from pypdf.generic import ArrayObject, BooleanObject, NullObject, NumberObject
    pdf, regions, _ = inventory_pdf
    writer = PdfWriter(clone_from=open_pdf(pdf))
    # Late failure after valid owned/legacy/foreign entries were already visited.
    last = writer.pages[1]['/Annots'][0].get_object()
    if change.startswith('duplicate'):
        if change == 'duplicate-legacy':
            ref = writer.pages[0]['/Annots'][2]
        else:
            ref = writer.pages[0]['/Annots'][0]
            if change == 'duplicate-dictionary':
                ref = writer._add_object(DictionaryObject(dict(ref.get_object())))
        writer.pages[1]['/Annots'].append(ref)
    elif change in ['short-id', 'uppercase-hex', 'trailing-space']:
        last.pop('/CiteHL')
        name = {'short-id': 'citehl:v1:bad', 'uppercase-hex': 'citehl:v1:' + 'A' * 64,
                'trailing-space': annotation_id(regions[1]) + ' '}[change]
        last[NameObject('/NM')] = TextStringObject(name)
    elif change == 'metadata-json':
        last[NameObject('/CiteHL')] = TextStringObject('{bad')
    elif change == 'metadata-type':
        last[NameObject('/CiteHL')] = NumberObject(1)
    elif change in ['renamed', 'no-nm']:
        last.pop('/NM')
        if change == 'renamed':
            last[NameObject('/NM')] = TextStringObject('foreign')
    elif change.startswith('metadata'):
        data = json.loads(last['/CiteHL'])
        if change == 'metadata-version':
            data['schema_version'] = True
        elif change == 'metadata-id':
            data['annotation_id'] = annotation_id(regions[0])
        elif change == 'metadata-region':
            data['region']['text'] = 'changed'
        elif change == 'metadata-extra':
            data['extra'] = 1
        raw = json.dumps(data)
        if change == 'metadata-duplicate-key':
            raw = raw[:-1] + ', "schema_version": 1}'
        last[NameObject('/CiteHL')] = TextStringObject(raw)
    elif change == 'missing-rect':
        last.pop('/Rect')
    elif change == 'short-rect':
        last[NameObject('/Rect')] = ArrayObject([NumberObject(1)])
    elif change == 'string-rect':
        last[NameObject('/Rect')] = TextStringObject('[1,2,3,4]')
    elif change == 'boolean-rect':
        last[NameObject('/Rect')][0] = BooleanObject(True)
    elif change == 'inverted-rect':
        last[NameObject('/Rect')] = RectangleObject((30, 20, 10, 10))
    elif change == 'null-annots':
        writer.pages[1][NameObject('/Annots')] = NullObject()
    elif change == 'dict-annots':
        writer.pages[1][NameObject('/Annots')] = DictionaryObject()
    elif change == 'nondict-entry':
        writer.pages[1]['/Annots'].append(NumberObject(1))
    writer.write(pdf)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert main(['list-annotations', str(pdf)]) == 1
    captured = capsys.readouterr()
    assert not captured.out and captured.err.startswith('ragwrench highlight: ')
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('kind', ['encrypted', 'userunit', 'rotation', 'malformed', 'missing'])
def test_unsupported_pdf(inventory_pdf, capsys, kind):
    from pypdf.generic import NumberObject
    pdf, _, _ = inventory_pdf
    writer = PdfWriter(clone_from=open_pdf(pdf))
    if kind == 'encrypted':
        writer.encrypt('')
    elif kind == 'userunit':
        writer.pages[1][NameObject('/UserUnit')] = NumberObject(2)
    elif kind == 'rotation':
        writer.pages[1][NameObject('/Rotate')] = NumberObject(45)
    writer.write(pdf)
    if kind == 'malformed':
        pdf.write_bytes(b'not PDF')
    elif kind == 'missing':
        pdf.unlink()
    before = pdf.read_bytes() if pdf.exists() else None
    assert main(['list-annotations', str(pdf)]) == 1
    captured = capsys.readouterr()
    assert not captured.out and captured.err.startswith('ragwrench highlight: ')
    assert (pdf.read_bytes() if pdf.exists() else None) == before


@pytest.mark.parametrize('extra', [[], ['x.pdf', '-o', 'out.pdf'], ['x.pdf', '--output', 'out.pdf'],
                                 ['x.pdf', '--force'], ['x.pdf', 'selection.json'],
                                 ['x.pdf', '--json'], ['x.pdf', '--text-check', 'off'],
                                 ['x.pdf', '--allow-empty'], ['x.pdf', '--tolerance', '1']])
def test_usage_rejects_write_and_other_options(extra, capsys):
    with pytest.raises(SystemExit) as exc:
        main(['list-annotations'] + extra)
    assert exc.value.code == 2
    assert not capsys.readouterr().out


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['list-annotations', '--help'])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert 'usage: ragwrench highlight list-annotations' in captured.out and not captured.err


@pytest.mark.parametrize('kind', ['pdf-bytes', 'owned-count', 'metadata-per-payload', 'metadata-total'])
def test_resource_limits(inventory_pdf, monkeypatch, capsys, kind):
    pdf, _, _ = inventory_pdf
    reader = open_pdf(pdf)
    payloads = [ref.get_object()['/CiteHL'] for p in reader.pages for ref in p['/Annots']
                if '/CiteHL' in ref.get_object()]
    if kind == 'pdf-bytes':
        monkeypatch.setattr('ragwrench.highlight.core.MAX_PDF_BYTES', 1)
    elif kind == 'owned-count':
        monkeypatch.setattr('ragwrench.highlight.inventory.MAX_REGIONS', 2)
    elif kind == 'metadata-per-payload':
        monkeypatch.setattr('ragwrench.highlight.pdf.MAX_JSON_BYTES', len(payloads[0].encode('utf-8')) - 1)
    else:
        monkeypatch.setattr('ragwrench.highlight.inventory.MAX_JSON_BYTES',
                            sum(len(p.encode('utf-8')) for p in payloads) - 1)
    before = pdf.read_bytes()
    assert main(['list-annotations', str(pdf)]) == 1
    captured = capsys.readouterr()
    assert not captured.out and captured.err.startswith('ragwrench highlight: ')
    if kind == 'metadata-total':
        assert 'embedded metadata exceeds' in captured.err
    assert pdf.read_bytes() == before


def test_serialized_output_limit_includes_newline(inventory_pdf, monkeypatch, capsys):
    from ragwrench.highlight.core import canonical
    from ragwrench.highlight.inventory import annotation_inventory
    pdf, _, _ = inventory_pdf
    size = len(canonical(annotation_inventory(open_pdf(pdf))).encode('utf-8')) + 1
    monkeypatch.setattr('ragwrench.highlight.inventory.MAX_JSON_BYTES', size)
    assert main(['list-annotations', str(pdf)]) == 0
    assert len(capsys.readouterr().out.encode('utf-8')) == size
    monkeypatch.setattr('ragwrench.highlight.inventory.MAX_JSON_BYTES', size - 1)
    assert main(['list-annotations', str(pdf)]) == 1
    captured = capsys.readouterr()
    assert not captured.out and 'JSON output limit' in captured.err


def test_inventory_selection_then_removal_remaining_ids(inventory_pdf, tmp_path, capsys):
    pdf, regions, _ = inventory_pdf
    assert main(['list-annotations', str(pdf)]) == 0
    entries = json.loads(capsys.readouterr().out)['annotations']
    selection = tmp_path / 'selection.json'
    # Select a legacy ID directly from inventory, without region reconstruction.
    selection.write_text(json.dumps({'version': 1, 'annotation_ids': [entries[1]['annotation_id']]}))
    output = tmp_path / 'removed.pdf'
    assert main(['remove', str(pdf), str(selection), '-o', str(output)]) == 0
    assert 'removed 1 annotations; preserved 3' in capsys.readouterr().out
    before = output.read_bytes()
    assert main(['list-annotations', str(output)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [entry['annotation_id'] for entry in data['annotations']] == [
        entry['annotation_id'] for entry in entries if entry != entries[1]]
    assert data['summary'] == {'owned': 2, 'foreign': 1}
    assert output.read_bytes() == before


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_nonfinite_current_rect_fails_without_partial_stdout(inventory_pdf, capsys, monkeypatch, value):
    from pypdf.generic import FloatObject
    pdf, _, _ = inventory_pdf
    reader = open_pdf(pdf)
    reader.pages[1]['/Annots'][0].get_object()['/Rect'][0] = FloatObject(value)
    monkeypatch.setattr('ragwrench.highlight.cli.open_pdf', lambda path: reader)
    assert main(['list-annotations', str(pdf)]) == 1
    captured = capsys.readouterr()
    assert not captured.out and 'finite' in captured.err
