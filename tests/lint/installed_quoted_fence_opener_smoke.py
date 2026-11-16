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

    a, b = '> # Example\n> ``', '`python\n> print("unique")\n> ```\n'
    cases = [('exact-requested', a, b, a + b, True)]
    for n in (3, 4, 5, 8):
        for char in ('`', '~'):
            marker = char * n
            for ending in ('\n', '\r\n', '\r'):
                prefix = '> ' if ending == '\n' else '   > '
                rows = ['# Café 雪', marker + 'python', '| before | table |', '| -- | -- |', '# heading']
                for length in range(1, n):
                    rows.extend([marker[:length], '| after | table |', '| -- | -- |', '# heading after'])
                rows.extend(['> ```literal', 'print("雪")\t ', marker + ' \t'])
                text = ending.join(prefix + line for line in rows) + ending
                offset = text.index(marker) + (1 if ending == '\r' else 2)
                cases.append((f'opaque-{n}-{ord(char)}-{len(cases)}', text[:offset], text[offset:], text, True))
    # Eight-character opener suffix remains unclosed on initial check for every
    # refusal below. Declare exits explicitly rather than deriving an oracle.
    a = '> # Example\n> ``'
    b = '``````python\n> unique payload\n> ````````\n'
    text = a + b
    closer = '> ````````\n'
    cases.extend([
        ('no-source', a, b, None, False),
        ('missing-byte', a, b[1:], text, False),
        ('changed-byte', a, b.replace('unique', 'changed'), text, False),
        ('changed-whitespace', a, b, text.replace('payload', 'payload '), False),
        ('missing-closer', a, b[:-len(closer)], text[:-len(closer)], False),
        ('short-closer', a, b.replace(closer, '> ```\n'), text.replace(closer, '> ```\n'), False),
        ('wrong-closer', a, b.replace(closer, '> ~~~~~~~~\n'), text.replace(closer, '> ~~~~~~~~\n'), False),
        ('invalid-closer', a, b.replace(closer, '> ```````` extra\n'), text.replace(closer, '> ```````` extra\n'), False),
        ('truncated-closer', a, b[:-2], text, False),
        ('ambiguous-with-hints', a, b, text + '\n' + text, False),
        ('literal-code', a, b, '> ~~~~~~~~~\n' + text + '> ~~~~~~~~~\n', False),
        ('source-closer', a, '``````\n> after\n> ````````\n> x\n> ````````\n',
         '> ````````\n' + a + '``````\n> after\n> ````````\n> x\n> ````````\n', False),
        ('overlap', a, text[len(a)-1:], text, False),
        ('reversal', a, b, b + '\n' + a, False),
    ])
    for name, transform in [
        ('nested', lambda s: s.replace('> ', '> > ')),
        ('mixed-prefix', lambda s: s.replace('> unique', ' > unique')),
        ('missing-prefix', lambda s: s.replace('> unique', 'unique')),
        ('list-preamble', lambda s: s.replace('> # Example', '> - item')),
        ('tab-prefix', lambda s: s.replace('> ', '>\t')),
        ('no-space-prefix', lambda s: s.replace('> ', '>')),
    ]:
        changed = transform(text); offset = changed.index('````````') + 2
        cases.append((name, changed[:offset], changed[offset:], changed, False))
    # A quoted three-marker interior line opens a shorter fence in a four-
    # marker right fragment cut after two markers; the final line closes it.
    clean_before = {f'opaque-4-{ord(char)}-{i}' for char, i in [('`', 7), ('`', 8), ('~', 10), ('~', 11)]}
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
