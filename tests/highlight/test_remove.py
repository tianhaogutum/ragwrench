"""Selective removal contract: strict v1 selection; exact persisted IDs only.

removed counts selected page-annotation entries; preserved counts all remaining
entries, including foreign and unselected owned annotations. No partial success.
Legacy owned /NM IDs may lack metadata. Present selected metadata must validate;
unselected annotation payloads/styles are preserved without re-verification.
"""
import json
from dataclasses import replace

import pytest
from pypdf import PdfWriter
from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject, IndirectObject, NameObject,
                           NumberObject, RectangleObject, TextStringObject)

from ragwrench.highlight.cli import main
from ragwrench.highlight.core import open_pdf, read_regions
from ragwrench.highlight.pdf import add_highlight, annotation_id, contents
from ragwrench.highlight.sidecar import _ObjectHasher, document_fingerprint


@pytest.fixture
def mixed(examples, tmp_path):
    regions = read_regions(examples / 'regions.jsonl')
    writer = PdfWriter(clone_from=open_pdf(examples / 'sample.pdf'))
    for r in regions:
        add_highlight(writer, r)
    appearance = DecodedStreamObject()
    appearance.set_data(b'q 1 0 0 rg 1 1 9 9 re f Q')
    writer.add_annotation(0, DictionaryObject({
        NameObject('/Subtype'): NameObject('/Highlight'),
        NameObject('/NM'): TextStringObject('foreign'),
        NameObject('/T'): TextStringObject('citehl'),
        NameObject('/Contents'): TextStringObject(contents(regions[0])),
        NameObject('/Rect'): RectangleObject(regions[0].rect),
        NameObject('/AP'): DictionaryObject({NameObject('/N'): writer._add_object(appearance)}),
        NameObject('/Custom'): DictionaryObject({NameObject('/Keep'): TextStringObject('原文')}),
    }))
    pdf = tmp_path / 'mixed.pdf'
    writer.write(pdf)
    return regions, pdf


def select(tmp_path, ids):
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps({'version': 1, 'annotation_ids': ids}))
    return path


def args(pdf, selection, output):
    return ['remove', str(pdf), str(selection), '-o', str(output)]


def snapshots(reader):
    # Hash complete retained dictionaries, resolving indirect appearance/custom
    # objects; omit only the page backlink, which contains the changed /Annots.
    return [[_ObjectHasher().hash(DictionaryObject({k: v for k, v in ref.get_object().items()
                                                   if k != '/P'}))
             for ref in page.get('/Annots', [])] for page in reader.pages]


def test_remove_exact_selected_preserves_owned_foreign_and_content(mixed, tmp_path, capsys):
    regions, pdf = mixed
    selection = select(tmp_path, [annotation_id(regions[0])])
    output = tmp_path / 'removed.pdf'
    before = pdf.read_bytes(), selection.read_bytes()
    reader = open_pdf(pdf)
    old = snapshots(reader)
    assert main(args(pdf, selection, output)) == 0
    assert capsys.readouterr().out == f'OK removed 1 annotations; preserved 2 -> {output}\n'
    result = open_pdf(output)
    assert snapshots(result) == [[old[0][1]], old[1]]
    assert document_fingerprint(result) == document_fingerprint(reader)
    assert [p.extract_text() for p in result.pages] == [p.extract_text() for p in reader.pages]
    assert (pdf.read_bytes(), selection.read_bytes()) == before


@pytest.mark.parametrize('chosen', [[], [0], [1], [1, 0]])
def test_empty_subset_all_and_request_order(mixed, tmp_path, capsys, chosen):
    regions, pdf = mixed
    selection = select(tmp_path, [annotation_id(regions[i]) for i in chosen])
    before = pdf.read_bytes(), selection.read_bytes()
    output = tmp_path / 'out.pdf'
    assert main(args(pdf, selection, output)) == 0
    assert capsys.readouterr().out == f'OK removed {len(chosen)} annotations; preserved {3-len(chosen)} -> {output}\n'
    old = snapshots(open_pdf(pdf))
    expected = [[old[0][0]] if 0 not in chosen else [], [old[1][0]] if 1 not in chosen else []]
    expected[0].append(old[0][1])
    assert snapshots(open_pdf(output)) == expected
    assert document_fingerprint(open_pdf(output)) == document_fingerprint(open_pdf(pdf))
    assert (pdf.read_bytes(), selection.read_bytes()) == before


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_mixed_same_page_rotations_and_untouched_pages(mixed, tmp_path, rotation):
    regions, pdf = mixed
    writer = PdfWriter(clone_from=open_pdf(pdf))
    writer.pages[0].rotate(rotation)
    # Another owned annotation shares the selected annotation's page. Page 2 is
    # untouched and an extra blank page has no /Annots entry at all.
    extra = replace(regions[0], chunk_id='same-page-other')
    add_highlight(writer, extra)
    writer.add_blank_page(200, 300)
    writer.write(pdf)
    old = snapshots(open_pdf(pdf))
    selection = select(tmp_path, [annotation_id(regions[0]), annotation_id(regions[1])])
    output = tmp_path / 'out.pdf'
    assert main(args(pdf, selection, output)) == 0
    result = open_pdf(output)
    assert snapshots(result) == [[old[0][1], old[0][2]], [], []]
    assert '/Annots' not in result.pages[2]
    assert result.pages[0].rotation == rotation
    assert document_fingerprint(result) == document_fingerprint(open_pdf(pdf))


@pytest.mark.parametrize('indirect', [False, True])
@pytest.mark.parametrize('kind', ['legacy', 'edited-style', 'unselected-bad-metadata', 'foreign-no-id'])
def test_only_selected_metadata_is_validated_and_no_appearance_ownership(mixed, tmp_path, kind, indirect):
    regions, pdf = mixed
    writer = PdfWriter(clone_from=open_pdf(pdf))
    first = writer.pages[0]['/Annots'][0].get_object()
    if indirect:
        first[NameObject('/NM')] = writer._add_object(first['/NM'])
    if kind == 'legacy':
        first.pop('/CiteHL')
    elif kind == 'edited-style':
        first[NameObject('/Contents')] = TextStringObject('edited by viewer')
        first[NameObject('/T')] = TextStringObject('viewer')
        first[NameObject('/Subtype')] = NameObject('/Square')
    elif kind == 'unselected-bad-metadata':
        writer.pages[1]['/Annots'][0].get_object()[NameObject('/CiteHL')] = TextStringObject('{bad')
    else:
        writer.pages[0]['/Annots'][1].get_object().pop('/NM')
    writer.write(pdf)
    old = snapshots(open_pdf(pdf))
    output = tmp_path / 'out.pdf'
    assert main(args(pdf, select(tmp_path, [annotation_id(regions[0])]), output)) == 0
    assert snapshots(open_pdf(output)) == [[old[0][1]], old[1]]


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('change', ['duplicate-request', 'missing', 'foreign', 'prefix', 'space', 'uppercase',
                                   'nonstring', 'null-ids', 'object-ids', 'string-ids', 'missing-ids',
                                   'missing-version', 'bool-version', 'float-version', 'future-version',
                                   'extra-field', 'root-array', 'duplicate-key', 'escaped-duplicate-key',
                                   'invalid-json', 'nonfinite'])
def test_invalid_selection_is_atomic(mixed, tmp_path, capsys, change, existing):
    regions, pdf = mixed
    name = annotation_id(regions[0])
    data = {'version': 1, 'annotation_ids': [name]}
    replacements = {
        'duplicate-request': [name, name], 'missing': [name, 'citehl:v1:' + '0'*64],
        'foreign': [name, 'foreign'], 'prefix': ['citehl:v1:'], 'space': [name + ' '],
        'uppercase': [name.upper()], 'nonstring': [True], 'null-ids': None,
        'object-ids': {}, 'string-ids': name,
    }
    if change in replacements:
        data['annotation_ids'] = replacements[change]
    elif change == 'missing-ids':
        data.pop('annotation_ids')
    elif change == 'missing-version':
        data.pop('version')
    elif change in ['bool-version', 'float-version', 'future-version']:
        data['version'] = {'bool-version': True, 'float-version': 1.0, 'future-version': 2}[change]
    elif change == 'extra-field':
        data['all'] = True
    elif change == 'root-array':
        data = []
    selection = tmp_path / 'selection.json'
    raw = json.dumps(data)
    if change in ['duplicate-key', 'escaped-duplicate-key']:
        key = 'annotation_ids' if change == 'duplicate-key' else r'annotat\u0069on_ids'
        raw = raw[:-1] + ', "' + key + '": []}'
    elif change == 'invalid-json':
        raw = '{broken'
    elif change == 'nonfinite':
        raw = '{"version":1,"annotation_ids":[NaN]}'
    selection.write_text(raw)
    before = pdf.read_bytes(), selection.read_bytes()
    output = tmp_path / 'out.pdf'
    if existing:
        output.write_bytes(b'existing output')
    assert main(args(pdf, selection, output) + ['--force']) == 1
    captured = capsys.readouterr()
    assert not captured.out and captured.err.startswith('ragwrench highlight: ')
    assert (pdf.read_bytes(), selection.read_bytes()) == before
    assert output.read_bytes() == b'existing output' if existing else not output.exists()
    assert not list(tmp_path.glob('.citehl-*'))


@pytest.mark.parametrize('indirect', [False, True])
@pytest.mark.parametrize('change', ['duplicate-selected-same-ref', 'duplicate-selected-new-object',
                                   'duplicate-unselected', 'duplicate-empty', 'metadata-json',
                                   'metadata-type', 'metadata-version', 'metadata-id', 'metadata-region',
                                   'metadata-extra', 'metadata-duplicate-key', 'renamed', 'no-nm'])
def test_ambiguous_ids_and_selected_metadata_fail_before_output(mixed, tmp_path, capsys, change, indirect, monkeypatch):
    regions, pdf = mixed
    writer = PdfWriter(clone_from=open_pdf(pdf))
    annotation = writer.pages[0]['/Annots'][0].get_object()
    if indirect:
        for page in writer.pages:
            owned = page['/Annots'][0].get_object()
            owned[NameObject('/NM')] = writer._add_object(owned['/NM'])
    ids = [annotation_id(regions[0])]
    if change.startswith('duplicate'):
        index = 1 if change in ['duplicate-unselected', 'duplicate-empty'] else 0
        ref = writer.pages[index]['/Annots'][0]
        if change == 'duplicate-selected-new-object':
            # Independent dictionary and across pages, not just repeated reference.
            ref = writer._add_object(DictionaryObject(dict(ref.get_object())))
        writer.pages[1]['/Annots'].append(ref)
        if change == 'duplicate-empty':
            ids = []
    elif change == 'metadata-json':
        annotation[NameObject('/CiteHL')] = TextStringObject('{bad')
    elif change == 'metadata-type':
        annotation[NameObject('/CiteHL')] = NumberObject(1)
    elif change == 'renamed':
        annotation[NameObject('/NM')] = TextStringObject('foreign-renamed')
    elif change == 'no-nm':
        annotation.pop('/NM')
    else:
        data = json.loads(annotation['/CiteHL'])
        if change == 'metadata-version':
            data['schema_version'] = True
        elif change == 'metadata-id':
            data['annotation_id'] = annotation_id(regions[1])
        elif change == 'metadata-region':
            data['region']['text'] = 'tampered'
        elif change == 'metadata-extra':
            data['extra'] = 1
        raw = json.dumps(data)
        if change == 'metadata-duplicate-key':
            raw = raw[:-1] + ', "schema_version": 1}'
        annotation[NameObject('/CiteHL')] = TextStringObject(raw)
    writer.write(pdf)
    before = pdf.read_bytes()
    selection = select(tmp_path, ids)
    output = tmp_path / 'out.pdf'
    output.write_bytes(b'sentinel')
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid targets must fail before cloning')
    monkeypatch.setattr('ragwrench.highlight.removal.PdfWriter', forbidden)
    assert main(args(pdf, selection, output) + ['--force']) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert ('duplicate' if change.startswith('duplicate') else
            'missing' if change in ['renamed', 'no-nm'] else 'metadata') in captured.err
    assert pdf.read_bytes() == before and output.read_bytes() == b'sentinel'
    assert not list(tmp_path.glob('.citehl-*'))


@pytest.mark.parametrize('pair', ['pdf-output', 'selection-output', 'pdf-selection'])
@pytest.mark.parametrize('alias', ['same', 'symlink', 'hardlink'])
def test_all_path_aliases_rejected(mixed, tmp_path, capsys, pair, alias):
    regions, pdf = mixed
    selection = select(tmp_path, [annotation_id(regions[0])])
    output = tmp_path / 'out.pdf'
    source = selection if pair == 'selection-output' else pdf
    target = source if alias == 'same' else tmp_path / 'alias'
    if alias == 'symlink':
        target.symlink_to(source)
    elif alias == 'hardlink':
        target.hardlink_to(source)
    if pair == 'pdf-selection':
        selection = target
    else:
        output = target
    before = pdf.read_bytes(), selection.read_bytes()
    assert main(args(pdf, selection, output) + ['--force']) == 1
    captured = capsys.readouterr()
    assert not captured.out and ('input' in captured.err or 'distinct' in captured.err)
    assert (pdf.read_bytes(), selection.read_bytes()) == before
    if pair == 'pdf-selection':
        assert not output.exists()


def test_overwrite_requires_force_and_repeat_removed_id_fails(mixed, tmp_path, capsys):
    regions, pdf = mixed
    selection = select(tmp_path, [annotation_id(regions[0])])
    output = tmp_path / 'out.pdf'
    output.write_bytes(b'sentinel')
    assert main(args(pdf, selection, output)) == 1
    assert output.read_bytes() == b'sentinel'
    assert main(args(pdf, selection, output) + ['--force']) == 0
    before = output.read_bytes()
    again = tmp_path / 'again.pdf'
    assert main(args(output, selection, again)) == 1
    assert output.read_bytes() == before and not again.exists()


@pytest.mark.parametrize('failure', ['serialize', 'replace'])
def test_failed_publication_preserves_existing_output(mixed, tmp_path, monkeypatch, failure):
    regions, pdf = mixed
    selection = select(tmp_path, [annotation_id(regions[0])])
    output = tmp_path / 'out.pdf'
    output.write_bytes(b'sentinel')
    before = pdf.read_bytes(), selection.read_bytes()
    if failure == 'serialize':
        def fail(self, stream):
            stream.write(b'partial')
            raise OSError('injected PDF serialization failure')
        monkeypatch.setattr(PdfWriter, 'write', fail)
    else:
        def fail(*args):
            raise OSError('injected atomic replacement failure')
        monkeypatch.setattr('ragwrench.highlight.core.os.replace', fail)
    assert main(args(pdf, selection, output) + ['--force']) == 1
    assert output.read_bytes() == b'sentinel'
    assert (pdf.read_bytes(), selection.read_bytes()) == before
    assert not list(tmp_path.glob('.citehl-*'))


@pytest.mark.parametrize('argv', [[], ['x.pdf'], ['x.pdf', 'selection.json'],
                                  ['x.pdf', 'selection.json', '-o', 'out.pdf', '--all']])
def test_usage_errors(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(['remove'] + argv)
    assert exc.value.code == 2
    assert not capsys.readouterr().out


def test_indirect_annotation_arrays(mixed, tmp_path):
    regions, pdf = mixed
    writer = PdfWriter(clone_from=open_pdf(pdf))
    for page in writer.pages:
        page[NameObject('/Annots')] = writer._add_object(page['/Annots'])
    writer.write(pdf)
    old = snapshots(open_pdf(pdf))
    output = tmp_path / 'out.pdf'
    assert main(args(pdf, select(tmp_path, [annotation_id(regions[0])]), output)) == 0
    assert snapshots(open_pdf(output)) == [[old[0][1]], old[1]]


def test_late_selected_metadata_failure_never_clones_or_writes(mixed, tmp_path, monkeypatch):
    regions, pdf = mixed
    writer = PdfWriter(clone_from=open_pdf(pdf))
    writer.pages[1]['/Annots'][0].get_object()[NameObject('/CiteHL')] = TextStringObject('{bad')
    writer.write(pdf)
    selection = select(tmp_path, [annotation_id(r) for r in regions])
    output = tmp_path / 'out.pdf'
    output.write_bytes(b'sentinel')
    def forbidden(*args, **kwargs):
        raise AssertionError('must validate the complete selection before cloning')
    monkeypatch.setattr('ragwrench.highlight.removal.PdfWriter', forbidden)
    assert main(args(pdf, selection, output) + ['--force']) == 1
    assert output.read_bytes() == b'sentinel'


@pytest.mark.parametrize('foreign', [False, True])
def test_empty_selection_without_owned_annotations(examples, tmp_path, foreign, capsys):
    writer = PdfWriter(clone_from=open_pdf(examples / 'sample.pdf'))
    if foreign:
        writer.add_annotation(0, DictionaryObject({
            NameObject('/Subtype'): NameObject('/Text'),
            NameObject('/Rect'): RectangleObject((1, 1, 10, 10)),
            NameObject('/T'): TextStringObject('citehl'),
            NameObject('/Contents'): TextStringObject('citehl:v1:' + '0'*64),
        }))
    pdf = tmp_path / 'source.pdf'
    writer.write(pdf)
    output = tmp_path / 'out.pdf'
    assert main(args(pdf, select(tmp_path, []), output)) == 0
    assert capsys.readouterr().out == f'OK removed 0 annotations; preserved {int(foreign)} -> {output}\n'
    assert snapshots(open_pdf(output)) == snapshots(open_pdf(pdf))
    assert document_fingerprint(open_pdf(output)) == document_fingerprint(open_pdf(pdf))


@pytest.mark.parametrize('kind', ['encrypted', 'userunit', 'malformed-pdf', 'annots-not-array', 'annot-not-dict'])
def test_unsupported_pdf_preserves_output(mixed, tmp_path, kind):
    regions, pdf = mixed
    writer = PdfWriter(clone_from=open_pdf(pdf))
    if kind == 'encrypted':
        writer.encrypt('')
    elif kind == 'userunit':
        writer.pages[0][NameObject('/UserUnit')] = NumberObject(2)
    elif kind == 'annots-not-array':
        writer.pages[0][NameObject('/Annots')] = DictionaryObject()
    elif kind == 'annot-not-dict':
        writer.pages[1]['/Annots'].append(NumberObject(1))
    writer.write(pdf)
    if kind == 'malformed-pdf':
        pdf.write_bytes(b'not a PDF')
    output = tmp_path / 'out.pdf'
    output.write_bytes(b'sentinel')
    assert main(args(pdf, select(tmp_path, [annotation_id(regions[0])]), output) + ['--force']) == 1
    assert output.read_bytes() == b'sentinel'


@pytest.mark.parametrize('limit', ['json-bytes', 'ids'])
def test_selection_resource_limits(mixed, tmp_path, monkeypatch, limit):
    regions, pdf = mixed
    selection = select(tmp_path, [annotation_id(regions[0])])
    if limit == 'json-bytes':
        monkeypatch.setattr('ragwrench.highlight.core.MAX_JSON_BYTES', 10)
    else:
        monkeypatch.setattr('ragwrench.highlight.removal.MAX_REGIONS', 0)
    output = tmp_path / 'out.pdf'
    output.write_bytes(b'sentinel')
    assert main(args(pdf, selection, output) + ['--force']) == 1
    assert output.read_bytes() == b'sentinel'


@pytest.mark.parametrize('moved', [False, True])
def test_remove_inventory_selected_indirect_id(mixed, tmp_path, capsys, moved):
    """Resolve persisted text identity; current page/rect never regenerate it."""
    regions, pdf = mixed
    writer = PdfWriter(clone_from=open_pdf(pdf))
    first = writer.pages[0]['/Annots'][0].get_object()
    first[NameObject('/NM')] = writer._add_object(first['/NM'])
    if moved:
        first[NameObject('/Rect')] = RectangleObject((1, 2, 30, 40))
        ref = writer.pages[0]['/Annots'].pop(0)
        writer.pages[1]['/Annots'].append(ref)
    writer.write(pdf)
    reader = open_pdf(pdf)
    page_index, array_index = (1, 1) if moved else (0, 0)
    selected = reader.pages[page_index]['/Annots'][array_index].get_object()
    assert isinstance(selected.get('/NM'), IndirectObject)
    assert main(['list-annotations', str(pdf)]) == 0
    inventory = json.loads(capsys.readouterr().out)
    entry = next(a for a in inventory['annotations'] if a['annotation_id'] == annotation_id(regions[0]))
    assert entry['metadata_available'] is True
    assert entry['region'] == regions[0].to_dict()
    selection = select(tmp_path, [entry['annotation_id']])
    before = pdf.read_bytes(), selection.read_bytes()
    old = snapshots(reader)
    output = tmp_path / 'out.pdf'
    assert main(args(pdf, selection, output)) == 0
    assert capsys.readouterr().out == f'OK removed 1 annotations; preserved 2 -> {output}\n'
    expected = [list(page) for page in old]
    expected[page_index].pop(array_index)
    result = open_pdf(output)
    assert snapshots(result) == expected
    assert document_fingerprint(result) == document_fingerprint(reader)
    assert [p.extract_text() for p in result.pages] == [p.extract_text() for p in reader.pages]
    assert (pdf.read_bytes(), selection.read_bytes()) == before
    assert isinstance(selected.get('/NM'), IndirectObject)  # Reader not mutated by removal.
    assert main(['list-annotations', str(output)]) == 0
    remaining = json.loads(capsys.readouterr().out)
    assert remaining['annotations'] == [a for a in inventory['annotations'] if a != entry]
    assert remaining['summary'] == {'owned': 1, 'foreign': 1}


@pytest.mark.parametrize('storage', ['direct-indirect', 'indirect-direct', 'indirect-shared-id',
                                    'indirect-separate-id', 'indirect-same-annotation'])
@pytest.mark.parametrize('chosen', ['duplicate', 'unrelated', 'empty'])
def test_indirect_duplicate_ids_globally_ambiguous(mixed, tmp_path, capsys, monkeypatch, storage, chosen):
    regions, pdf = mixed
    writer = PdfWriter(clone_from=open_pdf(pdf))
    first = writer.pages[0]['/Annots'][0].get_object()
    if storage != 'direct-indirect':
        first[NameObject('/NM')] = writer._add_object(first['/NM'])
    duplicate = DictionaryObject(dict(first))
    if storage in ['direct-indirect', 'indirect-separate-id']:
        duplicate[NameObject('/NM')] = writer._add_object(TextStringObject(annotation_id(regions[0])))
    elif storage == 'indirect-direct':
        duplicate[NameObject('/NM')] = TextStringObject(annotation_id(regions[0]))
    ref = (writer.pages[0]['/Annots'][0] if storage == 'indirect-same-annotation'
           else writer._add_object(duplicate))
    writer.pages[1]['/Annots'].append(ref)
    writer.write(pdf)
    selection = select(tmp_path, [] if chosen == 'empty' else [annotation_id(regions[chosen == 'unrelated'])])
    before = pdf.read_bytes(), selection.read_bytes()
    output = tmp_path / 'out.pdf'
    output.write_bytes(b'sentinel')
    def forbidden(*args, **kwargs):
        raise AssertionError('ambiguous IDs must fail before cloning')
    monkeypatch.setattr('ragwrench.highlight.removal.PdfWriter', forbidden)
    assert main(args(pdf, selection, output) + ['--force']) == 1
    captured = capsys.readouterr()
    assert not captured.out and 'ambiguous duplicate PDF annotation ID' in captured.err
    assert (pdf.read_bytes(), selection.read_bytes()) == before
    assert output.read_bytes() == b'sentinel' and not list(tmp_path.glob('.citehl-*'))


def test_indirect_integer_metadata_hash_and_reader_preserved(examples, tmp_path):
    from ragwrench.highlight.removal import remove_annotations
    region = replace(read_regions(examples / 'regions.jsonl')[0],
                     rect=(72, 686, 300, 700), text='原文 café', source_ref='a', source_refs=('a', 'b'))
    writer = PdfWriter(clone_from=open_pdf(examples / 'sample.pdf'))
    add_highlight(writer, region)
    first = writer.pages[0]['/Annots'][0].get_object()
    first[NameObject('/NM')] = writer._add_object(first['/NM'])
    pdf = tmp_path / 'integer.pdf'
    writer.write(pdf)
    reader = open_pdf(pdf)
    annotation = reader.pages[0]['/Annots'][0].get_object()
    raw_name = annotation.get('/NM')
    assert isinstance(raw_name, IndirectObject)
    old = snapshots(reader)
    output = tmp_path / 'out.pdf'
    assert remove_annotations(reader, {'version': 1, 'annotation_ids': [annotation_id(region)]}, output) == {
        'removed': 1, 'preserved': 0}
    assert annotation.get('/NM') is raw_name and snapshots(reader) == old
    assert snapshots(open_pdf(output)) == [[], []]
