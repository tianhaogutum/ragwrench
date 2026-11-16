"""Run with an installed wheel's Python and a durable evidence directory."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

import ragwrench.lint as chunklint


def main():
    assert 'site-packages' in str(chunklint.__file__), chunklint.__file__
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cli = Path(sys.executable).with_name('ragwrench')
    invocations = []

    def run(*args, expected):
        argv = [str(cli), 'lint', *map(str, args)]
        p = subprocess.run(argv, cwd=root, capture_output=True, timeout=20)
        stem = root / f'command-{len(invocations) + 1:03}'
        stem.with_suffix('.stdout').write_bytes(p.stdout)
        stem.with_suffix('.stderr').write_bytes(p.stderr)
        record = {'command': shlex.join(argv), 'argv': argv, 'cwd': str(root),
                  'exitCode': p.returncode, 'expectedExitCode': expected,
                  'rawLogPath': str(stem.with_suffix('.stdout')),
                  'stderrLogPath': str(stem.with_suffix('.stderr'))}
        invocations.append(record)
        (root / 'commands.json').write_text(json.dumps(invocations, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        assert p.returncode == expected, (p.stdout, p.stderr)
        assert not p.stderr
        return json.loads(p.stdout)

    a = '> | k | v |\n> | -- | -- |\n> | first | row |\n> | al'
    b = 'pha | b |\n'
    table = a + b
    cases = [('exact-requested', a, b, table, True)]
    for preceding in (1, 2, 7, 31):
        for ending in ('\n', '\r\n', '\r'):
            for prefix in ('> ', '   > '):
                rows = ['| k | v |', '| - | :-: |']
                rows += [f'| row {i} | preceding {i} |' for i in range(preceding)]
                rows += ['| café \\| 雪 | b |\t ', '| final | row |']
                text = ending.join(prefix + row for row in rows) + ending
                cut = text.index('\\|') + 1
                cases.append((f'raw-{preceding}-{len(cases)}', text[:cut], text[cut:], text, True))
    cases.extend([
        ('no-source', a, b, None, False),
        ('missing-byte', a, b[1:], table, False),
        ('changed-byte', a, b.replace('pha', 'pXa'), table, False),
        ('changed-whitespace', a, b, table.replace('alpha', 'al pha'), False),
        ('ambiguous-with-hints', a, b, table + '\n' + table, False),
        ('literal-code', a, b, '```md\n' + table + '```\n', False),
        ('quoted-code', a, b, '> ```md\n' + table + '> ```\n', False),
        ('overlap', a, 'l' + b, table, False),
        ('reversal', a, b, b + '\n' + a, False),
        ('source-extra-row', a, b, table + '> | next | row |\n', False),
        ('source-list', a, b, '- item\n' + table, False),
        ('partial-final-row', a, b, table.rstrip('\n') + ' extra\n', False),
        ('prefix-only', a[:-5], ' | alpha | b |\n', table, False),
    ])
    for name, old, new in [
        ('nested', '> ', '> > '),
        ('mixed-prefix', '> | first', ' > | first'),
        ('missing-prefix', '> | first', '| first'),
        ('list-body', '> | first', '> - | first'),
        ('invalid-row', '> | first | row |', '> prose'),
        ('invalid-delimiter', '| -- | -- |', '| -x | -- |'),
        ('source-blank', '> | first', '\n> | first'),
    ]:
        left, right = a.replace(old, new), b.replace(old, new)
        cases.append((name, left, right, left + right, False))
    clean_before = set()
    requested = set(sys.argv[2:])
    if requested:
        assert requested <= {case[0] for case in cases}, requested
        cases = [case for case in cases if case[0] in requested]
    evidence = []
    for name, a, b, source, positive in cases:
        inputs = [{'id': 'a', 'text': a, 'metadata': {'part': 'left'}},
                  {'id': 'b', 'text': b, 'metadata': {'part': 'right'}}]
        if name not in ('exact-requested',):
            inputs[0]['metadata']['chunklint'] = {'operation': 'rejoin_table', 'inputs': [
                {'id': 'a0', 'metadata': {'part': 0}}, {'id': 'a1', 'metadata': {'label': '雪'}}]}
        if name != 'exact-requested':
            inputs[1]['metadata']['chunklint'] = {'operation': 'rejoin_table', 'inputs': [
                {'id': 'b0', 'metadata': {'part': 2}}, {'id': 'b1', 'metadata': {'part': 3}}]}
        if name.endswith('ambiguous-with-hints'):
            inputs[0].update(start_char_idx=0, end_char_idx=len(a))
            inputs[1].update(start_char_idx=len(a), end_char_idx=len(a + b))
        chunks = root / (name + '.jsonl')
        chunks.write_bytes(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in inputs).encode())
        flags = ['--min-chars', 0, '--fail-on', 'warning', '--format', 'json']
        if source is not None:
            source_path = root / (name + '.md')
            source_path.write_bytes(source.encode())
            flags.extend(['--source', source_path])
        before_exit = 0 if name in clean_before else 1
        before = run('check', chunks, *flags, expected=before_exit)
        fixed, twice = root / (name + '-fixed.jsonl'), root / (name + '-twice.jsonl')
        status = 0 if positive else before_exit
        run('fix', chunks, *flags, '-o', fixed, expected=status)
        after = run('check', fixed, *flags, expected=status)
        output = [json.loads(line) for line in fixed.read_bytes().splitlines()]
        changes = json.loads(Path(str(fixed) + '.changes.json').read_bytes())
        if positive:
            assert len(output) == 1 and output[0]['id'] == 'a'
            assert output[0]['text'].encode() == source.encode() == (a + b).encode()
            prior = inputs[0]['metadata'].get('chunklint', {}).get('inputs')
            assert output[0]['metadata'] == {**inputs[0]['metadata'], 'chunklint': {
                'operation': 'rejoin_table', 'inputs': [
                    *(prior if prior else [{'id': 'a', 'metadata': inputs[0]['metadata']}]),
                    *inputs[1]['metadata'].get('chunklint', {}).get('inputs',
                        [{'id': 'b', 'metadata': inputs[1]['metadata']}])]}}
            assert changes['changes'] == [{'operation': 'rejoin_table', 'chunk_ids': ['a', 'b'], 'output_id': 'a'}]
            assert not after['issues'] and all(changes['invariant'].values())
            assert before['stats']['source']['coverage'] == after['stats']['source']['coverage'] == 1
        else:
            assert output == inputs and not changes['changes'] and after == before
        run('fix', fixed, *flags, '-o', twice, expected=status)
        second = json.loads(Path(str(twice) + '.changes.json').read_bytes())
        assert fixed.read_bytes() == twice.read_bytes() and not second['changes']
        evidence.append({'case': name, 'input': inputs, 'source': source, 'output': output,
                         'changeLog': changes, 'secondChangeLog': second, 'idempotent': True})
        (root / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'result': 'passed', 'cases': len(cases), 'cliInvocations': len(invocations),
                      'installedModule': chunklint.__file__, 'evidence': str(root / 'evidence.json')}))


if __name__ == '__main__':
    main()
