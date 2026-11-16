"""Run with an installed wheel's Python; retain the full CLI transcript in stdout."""
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
    cases = {
        'quoted-one': ['> | k | v |\n', '> | - | :-: |\n', '> | a | b |\n'],
        'unquoted-two': ['| k | v |\n', '| :-- | --: |\n', '| a | b |\n'],
        'nested-two-crlf': ['> > | k \\| key | v |\r\n', '>> | :--: | -- |\r\n',
                            ' > > | café \\| 雪 | b |\r\n'],
    }
    with tempfile.TemporaryDirectory(prefix='chunklint-installed-short-delimiter-') as directory:
        root = Path(directory)

        def write_records(path, values):
            path.write_bytes((''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values)).encode())

        def records(path):
            return [json.loads(line) for line in path.read_bytes().decode().splitlines()]

        def run(*args, expected=0):
            argv = [str(cli), 'lint', *map(str, args)]
            result = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=20)
            print(json.dumps({'command': shlex.join(argv), 'argv': argv, 'cwd': str(root),
                              'exitCode': result.returncode, 'stdout': result.stdout,
                              'stderr': result.stderr}, ensure_ascii=False), flush=True)
            assert result.returncode == expected, (result.stdout, result.stderr)
            return json.loads(result.stdout)

        for name, lines in cases.items():
            source = root / (name + '.md')
            source_bytes = ''.join(lines).encode()
            source.write_bytes(source_bytes)
            complete, split = root / (name + '-complete.jsonl'), root / (name + '-split.jsonl')
            inputs = [{'id': f'row-{i}', 'text': line,
                       'metadata': {'row': i, 'origin': name, 'nested': {'label': f'雪-{i}'}}}
                      for i, line in enumerate(lines)]
            write_records(complete, [{'id': 'complete', 'text': ''.join(lines)}])
            write_records(split, inputs)
            flags = ['--source', source, '--min-chars', 0, '--fail-on', 'warning', '--format', 'json']
            clean = run('check', complete, *flags)
            assert clean['issues'] == [] and clean['stats']['source']['coverage'] == 1
            before = run('check', split, *flags, expected=1)
            assert before['stats']['by_rule'] == {'CL002': 3}
            assert before['stats']['source']['coverage'] == 1
            fixed = root / (name + '-fixed.jsonl')
            run('fix', split, *flags, '-o', fixed)
            output = records(fixed)
            assert len(output) == 1 and output[0]['text'].encode() == source_bytes
            assert output[0]['metadata']['chunklint']['inputs'] == [
                {'id': value['id'], 'metadata': value['metadata']} for value in inputs]
            log = json.loads(Path(str(fixed) + '.changes.json').read_bytes())
            assert [change['operation'] for change in log['changes']] == ['rejoin_table'] * 2
            assert all(log['invariant'].values()) and not log['after']['issues']
            assert log['after']['stats']['source']['coverage'] == 1
            after = run('check', fixed, *flags)
            assert after['issues'] == [] and after['stats']['source']['coverage'] == 1
            twice = root / (name + '-twice.jsonl')
            run('fix', fixed, *flags, '-o', twice)
            second_log = json.loads(Path(str(twice) + '.changes.json').read_bytes())
            assert twice.read_bytes() == fixed.read_bytes() and not second_log['changes']
            print(json.dumps({'case': name, 'source': source_bytes.decode(), 'inputRecords': inputs,
                              'outputRecords': output, 'changeLog': log, 'secondChangeLog': second_log,
                              'exactRawTable': True, 'metadataInputs': 3, 'coverage': 1,
                              'idempotent': True}, ensure_ascii=False), flush=True)
        print(json.dumps({'result': 'passed', 'cases': list(cases)}), flush=True)


if __name__ == '__main__':
    main()
