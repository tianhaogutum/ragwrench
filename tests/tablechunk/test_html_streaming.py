"""HTML stream contracts: full records, source accounting and atomic failures."""
import io
import json
from pathlib import Path

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from .test_cli_benchmark import cli

FIXTURES = Path(__file__).parent / 'fixtures'


def stream(text, **kwargs):
    return list(stream_document(io.BytesIO(text.encode()), input_format='html', **kwargs))


@pytest.mark.parametrize('mode,budget', [('markdown', 33), ('kv', 26), ('html', 123)])
@pytest.mark.parametrize('adapter', ['jsonl', 'langchain', 'llamaindex'])
def test_accounting_fixture_full_jsonl_parity(mode, budget, adapter):
    text = (FIXTURES / 'spans-accounting.html').read_text()
    kwargs = dict(mode=mode, max_tokens=budget, source='same.html')
    expected = split_document(text, input_format='html', **kwargs)
    actual = stream(text, **kwargs)
    serialize = lambda chunks: ''.join(json.dumps(c.as_dict(adapter), ensure_ascii=False) + '\n' for c in chunks)
    assert serialize(actual) == serialize(expected)
    assert len(actual) == 3
    cells = {key for c in actual for key in c.metadata['normalization']['source_cells']}
    assert len(cells) == 10


def test_explicit_contract():
    text = '<h2>Sales</h2>\n<table>\n<tr><th>A</th><th>B</th></tr>\n<tr><td rowspan="0">West</td><td></td></tr>\n<tr></tr>\n</table>'
    chunks = stream(text, mode='kv', max_tokens=12)
    assert [c.text for c in chunks] == ['Sales\nColumns: A; B\n\nA: West\nB: ', 'Sales\nColumns: A; B\n\nA: West\nB: ']
    assert [c.id for c in chunks] == ['t0-r0-1', 't0-r1-2']
    meta = chunks[1].metadata
    assert meta['table_line_end'] == 6 and meta['row_line_spans'] == [[4, 5]]
    norm = meta['normalization']
    assert norm['row_cell_ids'] == [['r1c0', None]]
    assert norm['source_cells']['r1c0']['resolved_rowspan'] == 2
    assert norm['source_cells']['r1c0']['column_start'] == 4
    assert norm['diagnostics'][0]['code'] == 'missing_cell'


@pytest.mark.parametrize('text', [
    '', '<h1>Empty</h1><h2>Content</h2><p>Lead. Last.</p><table><tr><td>x</td></tr></table>Tail<h2>End</h2>',
    '<table><tr><th>A</th></tr></table><table><tr><td>x</td><td>y</td></tr></table>',
    (FIXTURES / 'row-groups.html').read_text(),
    '<table><tr><th>A</th></tr><tr><td>' + 'words ' * 200 + '</td></tr></table>',
    '<script>ignored <table></script><p>汉字 &amp; café<br>End</p>\r\n<table>\r<tr><td>世界</td></tr>\r\n</table>',
])
@pytest.mark.parametrize('mode,budget', [('markdown', 45), ('kv', 45), ('html', 150)])
def test_other_html_parity(text, mode, budget):
    kwargs = dict(max_tokens=budget, mode=mode)
    assert [c.as_dict() for c in stream(text, **kwargs)] == [c.as_dict() for c in split_document(text, input_format='html', **kwargs)]


@pytest.mark.parametrize('bad,error', [
    ('<table><tr><td>x</td></tr>', 'unclosed HTML table'),
    ('<table><tr><td>x</tr></table>', 'mismatched'),
    ('<table><tr><td rowspan="2">x</td></tr></table>', 'r0c0: rowspan extends beyond table'),
    ('<table><tbody><tr><td rowspan="2">x</td></tr></tbody><tbody><tr><td>y</td></tr></tbody></table>', 'crosses row group'),
    ('<table><tr><td colspan="0">x</td></tr></table>', 'colspan must'),
    ('<script>unfinished', 'unclosed script/style'),
])
def test_late_failures_atomic(tmp_path, bad, error):
    path = tmp_path / 'bad.html'
    path.write_text('<table><tr><td>earlier</td></tr></table>' + bad)
    output = tmp_path / 'out.jsonl'
    output.write_text('existing')
    for args in [(), ('-o', output)]:
        result = cli('split', path, '--stream', *args)
        assert result.returncode == 2 and error in result.stderr
        assert result.stdout == '' and output.read_text() == 'existing'
        assert not list(tmp_path.glob('.tablechunk-*'))


def test_bounded_reads_split_tags_utf8_and_early_close():
    class TinyReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 65536
            return super().read(min(size, 1))
    text = '<h2>汉字</h2>\r\n<table><tr><td>世界 &amp; café</td></tr></table>'
    source = TinyReader(text.encode())
    iterator = stream_document(source, input_format='html')
    assert [c.as_dict() for c in iterator] == [c.as_dict() for c in split_document(text, input_format='html')]
    assert not source.closed
    source.seek(0)
    iterator = stream_document(source, input_format='html')
    next(iterator)
    iterator.close()
    assert not source.closed


def test_stream_cli_parity(tmp_path):
    path = FIXTURES / 'spans-accounting.html'
    for mode in ('markdown', 'kv', 'html'):
        for adapter in ('jsonl', 'langchain', 'llamaindex'):
            args = ('split', path, '--mode', mode, '--adapter', adapter, '--max-tokens', '150')
            expected = cli(*args)
            actual = cli(*args, '--stream')
            assert actual.returncode == expected.returncode == 0, actual.stderr
            assert actual.stdout == expected.stdout and actual.stderr == expected.stderr
    text = path.read_text()
    actual = cli('split', '-', '--stream', '--format', 'html', input=text)
    expected = cli('split', '-', '--format', 'html', input=text)
    assert actual.returncode == 0 and actual.stdout == expected.stdout


@pytest.mark.parametrize('bad,error', [
    ('<p>' + 'x' * (1024 * 1024 + 1) + '</p>', 'working block'),
    ('<table><tr><td>' + 'x' * (1024 * 1024 + 1) + '</td></tr></table>', 'working block'),
    ('<!--' + 'x' * (1024 * 1024 + 1), 'unfinished token'),
    ('<script>' + 'x' * (1024 * 1024 + 1), 'unfinished token'),
    ('<table><tr><td>' + '<i>' * 1024 + '</i>' * 1024 + '</td></tr></table>', 'nesting exceeds'),
    ('<table><tr><th>A</th></tr>' + '<tr><td>x</td><td>y</td></tr>' * 50000 + '</table>', '100000 cells'),
])
def test_html_working_limits(bad, error):
    with pytest.raises(TableChunkError, match=error):
        stream(bad)


@pytest.mark.parametrize('finish', ['close', 'exhaust', 'error', 'storage-error'])
def test_html_disk_cleanup_and_ownership(monkeypatch, finish):
    import sqlite3
    import ragwrench.tablechunk.html_streaming as module
    from ragwrench.tablechunk.html_storage import DiskList
    original = module.DiskStore
    opened = []

    def track(*args):
        store = original(*args)
        opened.append(store)
        return store

    monkeypatch.setattr(module, 'DiskStore', track)
    text = b'<p>Lead.</p><table><tr><td>x</td></tr></table>'
    if finish == 'error':
        text += b'<table>'
    if finish == 'storage-error':
        def fail(*args):
            raise sqlite3.OperationalError('simulated disk full')
        monkeypatch.setattr(DiskList, 'append', fail)
    source = io.BytesIO(text)
    iterator = stream_document(source, input_format='html')
    if finish in ('error', 'storage-error'):
        with pytest.raises(TableChunkError, match='unclosed HTML table|temporary storage'):
            list(iterator)
    elif finish == 'close':
        next(iterator)
        iterator.close()
    else:
        list(iterator)
    assert not source.closed and len(opened) == 1
    assert not Path(opened[0].directory.name).exists()
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        opened[0].db.execute('SELECT 1')


def test_disk_backing_and_synthetic_parity(monkeypatch):
    # Run only after the small accounting fixture parity has been established.
    import ragwrench.tablechunk.html_streaming as module
    from ragwrench.tablechunk.html_storage import DiskList, DiskMap, DiskNormalization
    original = module._StreamingParser.make_table
    observed = []

    def inspect(self):
        table = original(self)
        assert isinstance(self.rows, DiskList)
        assert isinstance(table.rows, DiskList)
        assert isinstance(table.row_lines, DiskList)
        assert isinstance(table.normalization, DiskNormalization)
        assert isinstance(table.normalization.source_cells, DiskMap)
        assert isinstance(table.normalization.diagnostics, DiskList)
        observed.append(len(table.rows))
        return table

    monkeypatch.setattr(module._StreamingParser, 'make_table', inspect)
    text = '<h2>Large</h2><table><tr><th>A</th><th>B</th></tr><tbody>'
    text += ''.join(f'<tr><td rowspan="0">G{i}</td><td>{i}</td></tr><tr><td>{i+1}</td></tr></tbody><tbody>' for i in range(0, 2000, 2))
    text += '</tbody></table>'
    actual = stream(text, mode='kv', max_tokens=30)
    expected = split_document(text, input_format='html', mode='kv', max_tokens=30)
    assert [c.as_dict() for c in actual] == [c.as_dict() for c in expected]
    assert observed == [2000]


def test_disk_ragged_diagnostic_filtering():
    # Exercise indexed row diagnostics plus table-wide/header/cell diagnostics.
    text = '<table><tr><th></th><th></th></tr>'
    text += '<tr><td>x<table><tr><td>inner</td></tr></table></td></tr>' * 100
    text += '</table>'
    kwargs = dict(max_tokens=24)
    assert [c.as_dict() for c in stream(text, **kwargs)] == [c.as_dict() for c in split_document(text, input_format='html', **kwargs)]


def test_unknown_stream_input_format():
    with pytest.raises(TableChunkError, match='unknown input format'):
        list(stream_document(io.BytesIO(b''), input_format='csv'))


def test_html_storage_open_failure(monkeypatch):
    import sqlite3
    import ragwrench.tablechunk.html_storage as module

    def fail(*args):
        raise sqlite3.OperationalError('simulated open failure')
    monkeypatch.setattr(module.sqlite3, 'connect', fail)
    with pytest.raises(TableChunkError, match='temporary storage: simulated open failure'):
        stream('<table><tr><td>x</td></tr></table>')


@pytest.mark.parametrize('text,error', [
    ('<table>' + '<tr><th>' + 'x' * 400000 + '</th></tr>' + '<tr><th>' + 'y' * 400000 + '</th></tr>' + '<tr><th>' + 'z' * 400000 + '</th></tr></table>', 'headers exceed'),
    ('<table><tr><td colspan="256">' + 'x' * 5000 + '</td></tr></table>', 'working record'),
])
def test_header_and_span_expansion_buffer_limits(text, error):
    with pytest.raises(TableChunkError, match=error):
        stream(text)


def test_late_html_utf8_failure(tmp_path):
    path = tmp_path / 'bad.html'
    path.write_bytes(b'<table><tr><td>x</td></tr></table>' * 1000 + b'\xff')
    output = tmp_path / 'existing.jsonl'
    output.write_text('existing')
    result = cli('split', path, '--stream', '-o', output)
    assert result.returncode == 2 and 'utf-8' in result.stderr
    assert result.stdout == '' and output.read_text() == 'existing'
    assert not list(tmp_path.glob('.tablechunk-*'))
