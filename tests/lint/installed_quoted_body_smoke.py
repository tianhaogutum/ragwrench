"""Invoke with an installed wheel's Python; print raw reproducible CLI evidence."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import ragwrench.lint as chunklint


def main():
    assert 'site-packages' in str(chunklint.__file__), chunklint.__file__
    cli = Path(sys.executable).with_name('ragwrench')
    print(json.dumps({'installedModule': chunklint.__file__}), flush=True)
    invocations = 0
    with tempfile.TemporaryDirectory(prefix='chunklint-quoted-body-') as directory:
        root = Path(directory)

        def run(*args, expected):
            nonlocal invocations
            argv = [str(cli), 'lint', *map(str, args)]
            result = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=20)
            invocations += 1
            print(json.dumps({'command': shlex.join(argv), 'argv': argv, 'cwd': str(root),
                              'exitCode': result.returncode, 'stdout': result.stdout,
                              'stderr': result.stderr}, ensure_ascii=False), flush=True)
            assert result.returncode == expected, (result.stdout, result.stderr)
            return json.loads(result.stdout)

        cases = []
        for ending in ['\n', '\r\n', '\r']:
            for prefix in ['> ', '   > ']:
                table = ending.join(prefix + line for line in
                                    ['| k | v |', '| -- | :-: |', '| café \\| 雪 | b |\t ']) + ending
                cut = table.index('雪')
                cases.append((f'quoted-{len(cases)}', table[:cut], table[cut:], table, True))
        left, right = '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n'
        cases.append(('exact-requested', left, right, left + right, True))
        short = '> | k | v |\r\n> | - | :-: |\r\n> | café \\| 雪 | b |\r\n'
        cut = short.index('\\|') + 1
        cases.append(('short-colon-escape-cut', short[:cut], short[cut:], short, True))
        cases.extend([
            ('no-source', left, right, None, False),
            ('nested', left.replace('> ', '> > '), right,
             (left + right).replace('> ', '> > '), False),
            ('missing-prefix', left.replace('\n> | al', '\n| al'), right,
             left.replace('\n> | al', '\n| al') + right, False),
            ('invalid-delimiter', left.replace('-- |', '-x |', 1), right,
             left.replace('-- |', '-x |', 1) + right, False),
            ('source-blank', left, right, (left + right).replace('\n> | al', '\n\n> | al'), False),
            ('ambiguous', left, right, left + right + '\n' + left + right, False),
            ('changed-prefix', left.replace('\n> | al', '\n > | al'), right,
             left.replace('\n> | al', '\n > | al') + right, False),
            ('missing-character', left, right, (left + right).replace('alpha', 'alphxa'), False),
            ('prefix-cut', left[:-5], ' | alpha | b |\n', left + right, False),
            ('later-body', left.replace('\n> | al', '\n> | first | row |\n> | al'), right,
             left.replace('\n> | al', '\n> | first | row |\n> | al') + right, True),
            ('fenced-source', left, right, '> ```md\n' + left + right + '> ```\n', False),
            ('count-mismatch', left.replace('| k | v |', '| k | v | z |'), right,
             left.replace('| k | v |', '| k | v | z |') + right, False),
        ])
        for name, left, right, source_text, recovered in cases:
            inputs = [
                {'id': 'h', 'text': left, 'metadata': {'origin': 'quoted.md', 'chunklint': {
                    'operation': 'rejoin_table', 'inputs': [
                        {'id': 'h0', 'metadata': {'part': 0}},
                        {'id': 'h1', 'metadata': {'part': 1, 'label': '雪'}}]}}},
                {'id': 'r', 'text': right, 'metadata': {'origin': 'quoted.md', 'part': 2}}]
            chunks, source = root / (name + '.jsonl'), root / (name + '.md')
            chunks.write_bytes(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in inputs).encode())
            flags = ['--min-chars', 0, '--fail-on', 'warning', '--format', 'json']
            if source_text is not None:
                source.write_bytes(source_text.encode())
                flags.extend(['--source', source])
            before = run('check', chunks, *flags, expected=1)
            fixed, twice = root / (name + '-fixed.jsonl'), root / (name + '-twice.jsonl')
            expected = 0 if recovered else 1
            run('fix', chunks, *flags, '-o', fixed, expected=expected)
            after = run('check', fixed, *flags, expected=expected)
            output = [json.loads(line) for line in fixed.read_bytes().splitlines()]
            log = json.loads(Path(str(fixed) + '.changes.json').read_bytes())
            if recovered:
                assert len(output) == 1 and output[0]['text'].encode() == source.read_bytes()
                assert output[0]['id'] == 'h'
                assert output[0]['metadata']['chunklint']['inputs'] == [
                    *inputs[0]['metadata']['chunklint']['inputs'],
                    {'id': 'r', 'metadata': inputs[1]['metadata']}]
                assert log['changes'] == [{'operation': 'rejoin_table', 'chunk_ids': ['h', 'r'], 'output_id': 'h'}]
                assert not after['issues'] and all(log['invariant'].values())
                assert before['stats']['source']['coverage'] == after['stats']['source']['coverage'] == 1
                assert ''.join(c['text'] for c in inputs).encode() == output[0]['text'].encode()
            else:
                assert output == inputs and not log['changes']
                assert after == before
            run('fix', fixed, *flags, '-o', twice, expected=expected)
            second = json.loads(Path(str(twice) + '.changes.json').read_bytes())
            assert twice.read_bytes() == fixed.read_bytes() and not second['changes']
            print(json.dumps({'case': name, 'source': source_text, 'inputRecords': inputs,
                              'outputRecords': output, 'changeLog': log, 'secondChangeLog': second,
                              'idempotent': True}, ensure_ascii=False), flush=True)
        print(json.dumps({'result': 'passed', 'cases': len(cases), 'cliInvocations': invocations}), flush=True)


if __name__ == '__main__':
    main()
