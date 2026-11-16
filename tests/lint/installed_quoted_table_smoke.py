"""Run with an installed wheel's Python; exercise the CLI outside the repo."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import ragwrench.lint as chunklint

FIXTURE = Path(__file__).parent / 'fixtures' / 'quoted_table'


def main():
    cli = Path(sys.executable).with_name('ragwrench')
    assert 'site-packages' in str(chunklint.__file__), chunklint.__file__
    with tempfile.TemporaryDirectory(prefix='chunklint-installed-quoted-table-') as directory:
        root = Path(directory)
        chunks, source = root / 'chunks.jsonl', root / 'source.md'
        for path in (chunks, source):
            shutil.copyfile(FIXTURE / path.name, path)
        commands = []

        def run(*args, expected=0):
            result = subprocess.run([str(cli), 'lint', *map(str, args)], cwd=root,
                                    capture_output=True, text=True, timeout=20)
            commands.append({'args': list(map(str, args)), 'exitCode': result.returncode})
            assert result.returncode == expected, (result.stdout, result.stderr)
            return result

        def records(path):
            return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

        flags = ['--source', source, '--min-chars', 0, '--fail-on', 'warning']
        before = json.loads(run('check', chunks, *flags, '--format', 'json', expected=1).stdout)
        assert before['stats']['by_rule'] == {'CL002': 5}
        fixed = root / 'fixed.jsonl'
        run('fix', chunks, *flags, '-o', fixed)
        output = records(fixed)
        assert len(output) == 1 and output[0]['text'] == source.read_text(encoding='utf-8')
        inputs = output[0]['metadata']['chunklint']['inputs']
        assert [item['metadata'] for item in inputs] == [c['metadata'] for c in records(chunks)]
        log = json.loads(Path(str(fixed) + '.changes.json').read_text())
        assert [c['operation'] for c in log['changes']] == ['rejoin_table'] * 4
        assert log['after']['stats']['source']['coverage'] == 1
        assert all(log['invariant'].values())
        after = json.loads(run('check', fixed, *flags, '--format', 'json').stdout)
        assert after['issues'] == [] and after['stats']['source']['coverage'] == 1
        twice = root / 'twice.jsonl'
        run('fix', fixed, *flags, '-o', twice)
        assert twice.read_bytes() == fixed.read_bytes()
        plan, applied = root / 'plan.json', root / 'applied.jsonl'
        run('plan', chunks, *flags, '-o', plan)
        run('preview', chunks, '--source', source, '--plan', plan)
        run('apply', chunks, '--source', source, '--plan', plan, '-o', applied)
        assert records(applied) == output
        wrong, unchanged = root / 'wrong-depth.jsonl', root / 'unchanged.jsonl'
        bad = [{'id': 'h', 'text': '> > | key | value |\n'},
               {'id': 'd', 'text': '> | --- | --- |\n> | one | two |\n'}]
        wrong.write_text(''.join(json.dumps(c) + '\n' for c in bad), encoding='utf-8')
        run('fix', wrong, '--min-chars', 0, '--fail-on', 'warning', '-o', unchanged, expected=1)
        assert [c['text'] for c in records(unchanged)] == [c['text'] for c in bad]
        assert not json.loads(Path(str(unchanged) + '.changes.json').read_text())['changes']
        print(json.dumps({'result': 'passed', 'installedModule': chunklint.__file__,
                          'exactRawTable': True, 'metadataInputs': len(inputs), 'coverage': 1,
                          'idempotent': True, 'savedPlanMatchesFix': True,
                          'wrongDepthPreserved': True, 'commands': commands}, indent=2))


if __name__ == '__main__':
    main()
