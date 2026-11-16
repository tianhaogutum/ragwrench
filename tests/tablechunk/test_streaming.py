import io

import pytest

import ragwrench.tablechunk as tablechunk
from ragwrench.tablechunk import TableChunkError, split_document
from .test_cli_benchmark import cli


def streamed(text, **kwargs):
    return [chunk.as_dict() for chunk in tablechunk.stream_document(io.BytesIO(text.encode()), **kwargs)]


def test_explicit_stream_example():
    chunks = streamed('| A |\n|---|\n| x |\n| y |', max_tokens=12)
    assert [(c['id'], c['text'], c['metadata']['row_start'], c['metadata']['row_end']) for c in chunks] == [
        ('t0-r0-1', '| A |\n| --- |\n| x |', 0, 1),
        ('t0-r1-2', '| A |\n| --- |\n| y |', 1, 2),
    ]
    assert [c['metadata']['row_line_spans'] for c in chunks] == [[[3, 3]], [[4, 4]]]
    assert all(c['metadata']['table_line_end'] == 4 for c in chunks)


@pytest.mark.parametrize('mode,budget', [('markdown', 45), ('kv', 45), ('html', 135)])
@pytest.mark.parametrize('text', [
    '', '# Root\n## Empty\n## Content\n\nBody\n\n## Trailing',
    '# Root\n\nFirst.\n\nSecond.\n\n' + 'Sentence words. ' * 100 + '\n\nTail.',
    '# Catalog\n\nRates.\n\n| ID | Value |\n|---|---|\n' + '\n'.join(f'| R{i} | {i} |' for i in range(50)),
    'A | B\n:---|---:\nx|y|z\nq|r\n\nEnd.',
    '| A | B |\n|---|---|',
    '| A | B |\n|---|---|\n| ' + '汉字 escaped \\| value ' * 20 + '|',
    '```markdown\n<table>\n| A |\n|---|\n| x |\n```\n\n# Tail',
    'p\r\nq\r\n\r\n| `a|b` | X |\r\n|---|---|\r\n| a\\|b | c |\r\n',
    'p\u2028q\x85\x0b| A |\x0c|---|\r| x |\r\n',
], ids=[str(i) for i in range(10)])
def test_stream_full_record_parity(text, mode, budget):
    kwargs = dict(mode=mode, max_tokens=budget, source='same.md')
    assert streamed(text, **kwargs) == [c.as_dict() for c in split_document(text, **kwargs)]


def test_cli_stream_exact_bytes_all_adapters(tmp_path):
    path = tmp_path / 'input.md'
    path.write_text('# T\n\nA lead.\n\n| A | B |\n|---|---|\n| x |\n| y | z | extra |\n\nTail.')
    for mode in ('markdown', 'kv', 'html'):
        for adapter in ('jsonl', 'langchain', 'llamaindex'):
            args = ('split', path, '--mode', mode, '--adapter', adapter, '--max-tokens', '160')
            expected = cli(*args)
            output = tmp_path / 'out.jsonl'
            actual = cli(*args, '--stream', '-o', output)
            assert actual.returncode == expected.returncode == 0, actual.stderr
            assert output.read_text() == expected.stdout
            assert actual.stderr == expected.stderr
    stdin = cli('split', '-', '--stream', input=path.read_text())
    buffered = cli('split', '-', input=path.read_text())
    assert stdin.returncode == 0 and stdin.stdout == buffered.stdout


@pytest.mark.parametrize('bad,expected', [
    (b'\xff', 'utf-8'),
    (b'| A |\n|---|\n| x |\n\n<table>', 'embedded HTML'),
    (b'x' * (1024 * 1024 + 1), '1 MiB'),
    (b'```\n' + b'x\n' * (512 * 1024) + b'```', '1 MiB'),
], ids=['utf8', 'html', 'line-limit', 'block-limit'])
def test_late_cli_errors_are_atomic_and_cleanup(tmp_path, bad, expected):
    path = tmp_path / 'bad.md'
    # Force earlier chunks before the late failure, testing the output spool.
    path.write_bytes(b'| A |\n|---|\n' + b'| x |\n' * 30 + b'\n' + bad)
    output = tmp_path / 'output.jsonl'
    output.write_text('existing')
    for args in [('-o', output), ()]:
        run = cli('split', path, '--stream', '--max-tokens', '12', *args)
        assert run.returncode == 2 and expected in run.stderr
        assert run.stdout == ''
        assert output.read_text() == 'existing'
        assert not list(tmp_path.glob('.tablechunk-*'))


def test_stream_empty_html_table_rejected():
    run = cli('split', '-', '--stream', '--format', 'html', input='<table></table>')
    assert run.returncode == 2 and 'table contains no rows' in run.stderr and run.stdout == ''


def test_stream_bypasses_total_input_and_chunk_limits(monkeypatch):
    import ragwrench.tablechunk.splitter as splitter
    monkeypatch.setattr(splitter, 'MAX_INPUT_BYTES', 10)
    monkeypatch.setattr(splitter, 'MAX_CHUNKS', 2)
    text = '| A |\n|---|\n' + '| x |\n' * 30
    assert len(streamed(text, max_tokens=12)) == 30


def test_stream_limits_accumulation_with_zero_counter():
    text = ('x' * 300_000 + '\n\n') * 4
    with pytest.raises(TableChunkError, match='rendered buffer'):
        streamed(text, counter=lambda _: 0)


def test_stream_bounded_reads_and_early_close():
    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 65536
            return super().read(size)
    source = BoundedReader(b'| A |\n|---|\n' + b'| x |\n' * 1000)
    iterator = tablechunk.stream_document(source, max_tokens=12)
    assert next(iterator).metadata['row_start'] == 0
    iterator.close()
    assert not source.closed  # Caller owns the input; internal spools are closed.


def test_utf8_and_crlf_across_reads():
    class TinyReader(io.BytesIO):
        def read(self, size=-1):
            return super().read(1)
    text = '# 汉字\r\n\r\n| A |\r\n|---|\r\n| 世界 |\r\n'
    actual = list(tablechunk.stream_document(TinyReader(text.encode())))
    assert [c.as_dict() for c in actual] == [c.as_dict() for c in split_document(text)]


@pytest.mark.parametrize('finish', ['close', 'error', 'exhaust'])
def test_internal_spools_close(monkeypatch, finish):
    import ragwrench.tablechunk.streaming as streaming
    original = streaming.tempfile.TemporaryFile
    opened = []

    def track(*args, **kwargs):
        stream = original(*args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(streaming.tempfile, 'TemporaryFile', track)
    text = b'Before.\n\n| A |\n|---|\n' + b'| x |\n' * 30
    if finish == 'error':
        text += b'\n<table>'
    iterator = tablechunk.stream_document(io.BytesIO(text), max_tokens=18)
    next(iterator)
    if finish == 'close':
        iterator.close()
    elif finish == 'error':
        with pytest.raises(TableChunkError, match='embedded HTML'):
            list(iterator)
    else:
        list(iterator)
    assert len(opened) == 2 and all(stream.closed for stream in opened)


def test_seeded_mixed_document_parity():
    import random
    rng = random.Random(24)
    for mode, budget in [('markdown', 65), ('kv', 65), ('html', 180)]:
        for _ in range(12):
            parts = ['# Root', '## Empty', '## Content', 'A lead.\n\nAnother paragraph.']
            for index in range(3):
                parts += ['| A | B |\n|---|---|\n' + '\n'.join(
                    '| ' + ' | '.join(['R' + str(row), 'words ' * rng.randint(0, 25)] +
                                     (['extra'] if rng.randrange(3) == 0 else [])) + ' |'
                    for row in range(rng.randint(0, 20))), 'Tail.', f'## Section {index}']
            text = '\n\n'.join(parts)
            kwargs = dict(max_tokens=budget, mode=mode)
            assert streamed(text, **kwargs) == [c.as_dict() for c in split_document(text, **kwargs)]


def test_limit_sized_line_with_trailing_cr():
    text = 'x' * (1024 * 1024) + '\r'
    assert streamed(text) == [c.as_dict() for c in split_document(text)]
