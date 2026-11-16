"""Run with an installed wheel's Python; stdout retains exact CLI evidence."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import ragwrench.lint as chunklint


def main():
    cli = Path(sys.executable).with_name('ragwrench')
    assert 'site-packages' in str(chunklint.__file__), chunklint.__file__
    print(json.dumps({'installedModule': chunklint.__file__}), flush=True)
    with tempfile.TemporaryDirectory(prefix='chunklint-installed-table-boundary-') as directory:
        root = Path(directory)

        def run(*args, expected):
            argv = [str(cli), 'lint', *map(str, args)]
            result = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=20)
            print(json.dumps({'command': shlex.join(argv), 'argv': argv, 'cwd': str(root),
                              'exitCode': result.returncode, 'stdout': result.stdout,
                              'stderr': result.stderr}, ensure_ascii=False), flush=True)
            assert result.returncode == expected, (result.stdout, result.stderr)
            return json.loads(result.stdout)

        def records(path):
            return [json.loads(line) for line in path.read_bytes().decode().splitlines()]

        for name, ending, separator, token_cut in [
            ('blank-lf', '\n', '\n', False),
            ('blank-crlf', '\r\n', '\r\n', False),
            ('blank-cr', '\r', '\r', False),
            ('adjacent', '\n', '', False),
            ('token-cut', '\r\n', '', True),
            ('delimiter-cut-lf', '\n', '', 'delimiter'),
            ('delimiter-cut-crlf', '\r\n', '', 'delimiter'),
            ('delimiter-cut-cr', '\r', '', 'delimiter'),
            ('header-cut-lf', '\n', '', 'header'),
            ('header-cut-crlf-escaped', '\r\n', '', 'header-escaped'),
            ('header-cut-cr-bare', '\r', '', 'header-bare'),
            ('preamble-header-lf', '\n', '', 'preamble-header'),
            ('preamble-header-crlf-escaped', '\r\n', '', 'preamble-header-escaped'),
            ('preamble-header-cr-bare', '\r', '', 'preamble-header-bare'),
        ]:
            preamble = ''
            if isinstance(token_cut, str) and token_cut.startswith('preamble-'):
                preamble = ('Introduction.' + ending + ending if ending == '\n' else
                            '\t Résumé 雪.  ' + ending + 'Second line.' + ending + ' \t' + ending)
                token_cut = token_cut.removeprefix('preamble-')
            header = '| k | v |' + ending + '| - | :-: |' + ending
            if token_cut == 'delimiter':
                header = '| k | v |' + ending + '| -- | -- |' + ending
            elif isinstance(token_cut, str) and token_cut.startswith('header'):
                title = {'header': '| key | v |', 'header-escaped': '| café \\| 雪 | v |',
                         'header-bare': 'key | value'}[token_cut]
                header = title + ending + '| -- | -- |' + ending
            row = ('| café \\| 雪 | beta |' + ending
                   if token_cut is True else '| a | b |' + ending)
            source_text = header + separator + row
            if token_cut == 'delimiter':
                cut = header.index('--') + 1
                left, right = source_text[:cut], source_text[cut:]
            elif isinstance(token_cut, str) and token_cut.startswith('header'):
                cut = {'header': 4, 'header-escaped': 8, 'header-bare': 9}[token_cut]
                left, right = source_text[:cut], source_text[cut:]
            elif token_cut:
                cut = len(header) + row.index('café') + 3
                left, right = source_text[:cut], source_text[cut:]
            else:
                left, right = header, row
            source_text = preamble + source_text
            left = preamble + left
            inputs = [{'id': 'h', 'text': left, 'metadata': {'part': 'header', 'nested': {'label': '雪'}}},
                      {'id': 'r', 'text': right, 'metadata': {'part': 'body'}}]
            if separator:
                inputs[0].update(start_char_idx=0, end_char_idx=len(left))
                inputs[1].update(start_char_idx=len(header + separator), end_char_idx=len(source_text))
            source, chunks = root / (name + '.md'), root / (name + '.jsonl')
            source.write_bytes(source_text.encode())
            chunks.write_bytes((''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in inputs)).encode())
            flags = ['--source', source, '--min-chars', 0, '--fail-on', 'warning', '--format', 'json']
            before = run('check', chunks, *flags, expected=1)
            assert before['stats']['source']['coverage'] == 1
            fixed, twice = root / (name + '-fixed.jsonl'), root / (name + '-twice.jsonl')
            expected = 1 if separator else 0
            run('fix', chunks, *flags, '-o', fixed, expected=expected)
            after = run('check', fixed, *flags, expected=expected)
            log = json.loads(Path(str(fixed) + '.changes.json').read_bytes())
            output = records(fixed)
            if separator:
                assert output == inputs
                assert after['issues'] == before['issues']
                assert [(i['rule'], i['chunk_id'], i['line']) for i in after['issues']] == [('CL002', 'r', 1)]
                assert not log['changes']
                assert log['skipped'] == [{
                    'operation': 'rejoin_table', 'chunk_ids': ['h', 'r'],
                    'reason': 'recovered source does not have a contiguous table boundary'}]
            else:
                assert len(output) == 1 and output[0]['text'].encode() == source.read_bytes()
                assert output[0]['id'] == 'h'
                assert output[0]['metadata']['chunklint']['inputs'] == [
                    {'id': c['id'], 'metadata': c['metadata']} for c in inputs]
                assert [c['operation'] for c in log['changes']] == ['rejoin_table']
                assert not after['issues']
            assert after['stats']['source']['coverage'] == 1 and all(log['invariant'].values())
            run('fix', fixed, *flags, '-o', twice, expected=expected)
            second_log = json.loads(Path(str(twice) + '.changes.json').read_bytes())
            assert twice.read_bytes() == fixed.read_bytes() and not second_log['changes']
            print(json.dumps({'case': name, 'source': source_text, 'inputRecords': inputs,
                              'outputRecords': output, 'changeLog': log, 'secondChangeLog': second_log,
                              'coverage': 1, 'idempotent': True}, ensure_ascii=False), flush=True)
        print(json.dumps({'result': 'passed', 'cases': 14, 'cliInvocations': 56}), flush=True)


if __name__ == '__main__':
    main()
