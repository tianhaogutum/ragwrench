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

    left, right = '# Example\n``', "`python\nprint('x')\n```\n"
    cases = [('exact-requested', left, right, left + right, True)]
    for marker, ending, cut in [('```', '\r\n', 1), ('~~~', '\r', 2), ('~~~', '\n', 1)]:
        text = ending.join(['# Café 雪', '   ' + marker + 'python\t title="雪" ',
                            '| k | v |', '| -- | :-: |', '# heading', '- list',
                            '> ```literal', marker[:2], 'print("雪")\t ', '   ' + marker, ''])
        offset = text.index(marker) + cut
        cases.append((f'opaque-{len(cases)}', text[:offset], text[offset:], text, True))
    four_a, four_b = '# Example\n``', '``python\nprint("unique")\n````\n'
    cases.append(('four-exact-requested', four_a, four_b, four_a + four_b, True))
    for marker in ['````', '~~~~']:
        for ending in ['\n', '\r\n', '\r']:
            for cut in [1, 2]:
                text = ending.join(['# Café 雪', '   ' + marker + 'python\t title="雪" ',
                                    '| before | table |', '# heading before', marker[:3],
                                    '| after | table |', '# heading after', 'print("雪")\t ',
                                    '   ' + marker + ' \t', ''])
                offset = text.index(marker) + cut
                cases.append((f'four-opaque-{len(cases)}', text[:offset], text[offset:], text, True))
    cases.extend([
        ('four-no-source', four_a, four_b, None, False),
        ('four-missing-marker-byte', four_a, four_b[1:], four_a + four_b, False),
        ('four-missing-closer', four_a, four_b[:-5], four_a + four_b, False),
        ('four-short-closer', four_a, four_b[:-5] + '```\n', four_a + four_b[:-5] + '```\n', False),
        ('four-ambiguous-with-hints', four_a, four_b, four_a + four_b + '\n' + four_a + four_b, False),
        ('four-list-container', '- item\n  ``', '``python\n  x\n  ````\n',
         '- item\n  ````python\n  x\n  ````\n', False),
        ('four-literal-in-source-fence', four_a, four_b, '~~~~~\n' + four_a + four_b + '~~~~~\n', False),
    ])
    cases.extend([
        ('no-source', left, right, None, False),
        ('missing-marker-byte', left, right[1:], left + right, False),
        ('missing-closer', left, right[:-4], left + right, False),
        ('ambiguous-with-hints', left, right, left + right + '\n' + left + right, False),
        ('list-container', '- item\n  ``', '`python\n  x\n  ```\n',
         '- item\n  ```python\n  x\n  ```\n', False),
        ('literal-in-source-fence', left, right, '~~~~\n' + left + right + '~~~~\n', False),
    ])
    # A long suffix can be a complete fence by itself: check may exit 0
    # before this source-backed repair. Refusals can likewise be lint-clean.
    clean_before = {'long-exact-requested', 'long-no-source', 'long-short-closer', 'long-list-container'}
    long_a, long_b = '# Example\n``', '```python\nprint("unique")\n`````\n'
    cases.append(('long-exact-requested', long_a, long_b, long_a + long_b, True))
    for n in (5, 6, 8, 37):
        for char in ('`', '~'):
            marker = char * n
            cut = 1 if char == '~' else 2
            ending = {5: '\n', 6: '\r\n', 8: '\r', 37: '\r\n'}[n]
            text = ending.join(['# Café 雪', '   ' + marker + 'python\t title="雪" ',
                                'print("雪")\t ', '   ' + marker + ' \t', ''])
            offset = text.index(marker) + cut
            name = f'long-simple-{n}-{ord(char)}'
            clean_before.add(name)
            cases.append((name, text[:offset], text[offset:], text, True))
            text = ending.join(['# Café 雪', marker + 'python', '| before | table |',
                                '| -- | -- |', '# heading before', marker[:-1],
                                '| after | table |', '| -- | -- |', '# heading after',
                                'unique payload', marker, ''])
            offset = text.index(marker) + cut
            cases.append((f'long-opaque-{n}-{ord(char)}', text[:offset], text[offset:], text, True))
    cases.extend([
        ('long-no-source', long_a, long_b, None, False),
        ('long-missing-byte', long_a, long_b[1:], long_a + long_b, False),
        ('long-missing-closer', long_a, long_b[:-6], long_a + long_b, False),
        ('long-short-closer', long_a, long_b[:-6] + '````\n',
         long_a + long_b[:-6] + '````\n', False),
        ('long-wrong-closer', long_a, long_b[:-6] + '~~~~~\n',
         long_a + long_b[:-6] + '~~~~~\n', False),
        ('long-ambiguous-with-hints', long_a, long_b, long_a + long_b + '\n' + long_a + long_b, False),
        ('long-list-container', '- item\n  ``', '```python\n  x\n  `````\n',
         '- item\n  `````python\n  x\n  `````\n', False),
        ('long-literal-in-source-fence', long_a, long_b, '~~~~~~\n' + long_a + long_b + '~~~~~~\n', False),
    ])
    requested = set(sys.argv[2:])
    if requested:
        assert requested <= {case[0] for case in cases}, requested
        cases = [case for case in cases if case[0] in requested]
    evidence = []
    for name, a, b, source, positive in cases:
        inputs = [{'id': 'a', 'text': a, 'metadata': {'part': 'left'}},
                  {'id': 'b', 'text': b, 'metadata': {'part': 'right'}}]
        if name not in ('exact-requested', 'long-exact-requested'):
            inputs[0]['metadata']['chunklint'] = {'operation': 'rejoin_fence', 'inputs': [
                {'id': 'a0', 'metadata': {'part': 0}}, {'id': 'a1', 'metadata': {'label': '雪'}}]}
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
                'operation': 'rejoin_fence', 'inputs': [
                    *(prior if prior else [{'id': 'a', 'metadata': inputs[0]['metadata']}]),
                    {'id': 'b', 'metadata': inputs[1]['metadata']}]}}
            assert changes['changes'] == [{'operation': 'rejoin_fence', 'chunk_ids': ['a', 'b'], 'output_id': 'a'}]
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
