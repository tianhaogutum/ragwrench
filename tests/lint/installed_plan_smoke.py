"""Run with an installed wheel's Python: python tests/lint/installed_plan_smoke.py.

Creates its own synthetic snapshots outside the repo; no fixtures or network.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import ragwrench.lint as chunklint
from jsonschema import Draft202012Validator
from importlib.resources import files


def main():
    cli = Path(sys.executable).with_name('ragwrench')
    with tempfile.TemporaryDirectory(prefix='chunklint-installed-plan-') as directory:
        root = Path(directory)
        commands = []

        def run(*args, expected=0):
            result = subprocess.run([str(cli), 'lint', *map(str, args)], cwd=root, text=True,
                                    capture_output=True, timeout=20)
            commands.append({'args': list(map(str, args)), 'exitCode': result.returncode})
            assert result.returncode == expected, (result.stdout, result.stderr)
            return result

        chunks, source, plan = [root / name for name in ('input.json', 'source.md', 'plan.json')]
        chunks.write_text(json.dumps([{'id': 'h', 'text': '# Café'}, {'id': 'p', 'text': 'Body.'}]))
        source.write_text('# Café\nBody.', encoding='utf-8')
        run('plan', chunks, '--source', source, '--min-chars', 0, '-o', plan)
        original_chunks, original_source, original_plan = [p.read_bytes() for p in (chunks, source, plan)]
        chunks.unlink()
        source.unlink()
        (root / 'chunklint.toml').write_text('broken ambient config')
        before = {p.name: p.read_bytes() for p in root.iterdir()}
        valid = run('validate-plan', plan, '--format', 'json')
        assert json.loads(valid.stdout) == {'valid': True, 'version': 1, 'edits': 1,
                                           'snapshots_checked': False}
        assert not valid.stderr
        assert run('validate-plan', plan).stdout == 'Valid fix plan v1: 1 edits (snapshots not checked)\n'
        schema_result = run('validate-plan', '--schema')
        schema = json.loads(schema_result.stdout)
        packaged = json.loads(files('ragwrench.lint').joinpath('schemas', 'fix-plan-v1.schema.json').read_text())
        assert schema == packaged
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(json.loads(plan.read_text()))
        assert before == {p.name: p.read_bytes() for p in root.iterdir()}
        value = json.loads(plan.read_text())
        value['edits'][0]['replacement'][0]['text'] = False
        plan.write_text(json.dumps(value))
        invalid = run('validate-plan', plan, '--format', 'json', expected=2)
        data = json.loads(invalid.stdout)
        assert not invalid.stderr and data['valid'] is False
        assert data['diagnostics'][0]['path'] == '$.edits[0].replacement[0].text'

        output = root / 'out.jsonl'
        sentinel = b'KEEP\x00\xff\r\n'
        output.write_bytes(sentinel)

        def snapshot_files():
            return {p.name: p.read_bytes() for p in root.iterdir()}

        for unavailable in (chunks, source):
            chunks.write_bytes(original_chunks)
            source.write_bytes(original_source)
            unavailable.unlink()
            before = snapshot_files()
            for command in ('preview', 'apply'):
                options = ['-o', output, '--format', 'json'] if command == 'apply' else []
                rejected = run(command, chunks, '--source', source, '--plan', plan, *options, expected=2)
                assert not rejected.stdout
                assert rejected.stderr == ('ragwrench lint: invalid fix plan at '
                                            '$.edits[0].replacement[0].text: expected string\n')
                assert before == snapshot_files()

        chunks.write_bytes(original_chunks)
        source.write_bytes(original_source)
        plan.write_bytes(original_plan)
        run('preview', chunks, '--source', source, '--plan', plan)
        applied = run('apply', chunks, '--source', source, '--plan', plan, '-o', output, '--format', 'json')
        assert json.loads(applied.stdout)['invariant']['source_coverage_preserved']
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(records) == 1 and records[0]['text'] == '# Café\nBody.'
        assert hashlib.sha256(output.read_bytes()).hexdigest() == json.loads(plan.read_text())['output_sha256']
        assert not Path(str(output) + '.changes.json').exists()

        output.write_bytes(sentinel)
        for stale in (chunks, source):
            chunks.write_bytes(original_chunks)
            source.write_bytes(original_source)
            stale.write_bytes(stale.read_bytes() + b'\n')
            before = snapshot_files()
            for command in ('preview', 'apply'):
                options = ['-o', output] if command == 'apply' else []
                rejected = run(command, chunks, '--source', source, '--plan', plan, *options, expected=2)
                assert 'snapshot differs' in rejected.stderr and not rejected.stdout
                assert before == snapshot_files()

        chunks.write_bytes(original_chunks)
        source.write_bytes(original_source)
        value = json.loads(original_plan)
        value['edits'][0]['replacement'][0]['text'] = 'Structurally valid but not the saved repair.'
        plan.write_text(json.dumps(value))
        before = snapshot_files()
        rejected = run('apply', chunks, '--source', source, '--plan', plan, '-o', output, expected=2)
        assert 'do not match the deterministic repair' in rejected.stderr and not rejected.stdout
        assert before == snapshot_files()
        assert 'site-packages' in str(chunklint.__file__), chunklint.__file__
        print(json.dumps({'result': 'passed', 'installedModule': chunklint.__file__,
                          'noSnapshots': True, 'packagedSchema': True,
                          'malformedPlanPrecedesUnavailableSnapshots': True,
                          'validApply': True, 'staleSnapshotsRejected': True,
                          'deterministicRepairEnforced': True, 'sentinelPreserved': True,
                          'commands': commands}, indent=2))


if __name__ == '__main__':
    main()
