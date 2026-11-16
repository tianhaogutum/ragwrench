"""Run filtered checks and hit assertions using the installed CLI on loopback."""
import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path


def main():
    cli = str(Path(sys.executable).with_name('ragwrench'))
    config = Path(__file__).with_name('bridge.toml')
    env = dict(os.environ, BRIDGE_API_KEY='local-demo')
    process = subprocess.Popen([cli, 'dify-bridge', 'serve', '--config', str(config), '--port', '0'],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 5)[0], 'CLI startup timed out'
        line = process.stdout.readline()
        assert '127.0.0.1:' in line, 'CLI did not announce its loopback endpoint'
        endpoint = 'http://' + line.strip().split()[-1]
        base = [cli, 'dify-bridge', 'check', '--endpoint', endpoint, '--knowledge-id', 'handbook',
                '--query', 'vacation policy', '--top-k', '3', '--score-threshold', '0.2']
        condition = {'logical_operator': 'and', 'conditions': [
            {'name': 'lang', 'comparison_operator': 'is', 'value': 'en'}]}

        def check(flags, expected_code):
            result = subprocess.run(base + flags, env=env, capture_output=True, text=True, timeout=15)
            assert result.returncode == expected_code, result.stderr + result.stdout
            assert all(value not in result.stdout + result.stderr for value in
                       ['local-demo', 'vacation policy', 'handbook.md', 'leave.md', 'missing-title-marker'])
            if expected_code == 2:
                assert not result.stdout and 'Configuration' in result.stderr
                return
            report = json.loads(result.stdout)
            assert report['passed'] is (expected_code == 0)
            assert not result.stderr
            if expected_code == 0:
                assert len(report['checks']) == 13 and all(c['passed'] for c in report['checks'])
            return report

        filtered = ['--metadata-condition', json.dumps(condition)]
        check(filtered + ['--expect-count', '2', '--expect-title', 'handbook.md', '--expect-title', 'leave.md'], 0)
        mismatch = check(filtered + ['--expect-count', '1'], 1)
        assert 'baseline record count does not match --expect-count' in mismatch['checks'][0]['errors']
        mismatch = check(filtered + ['--expect-title', 'missing-title-marker', '--allow-empty'], 1)
        assert 'baseline is missing an --expect-title' in mismatch['checks'][0]['errors']
        condition['conditions'][0]['value'] = 'absent'
        check(['--metadata-condition', json.dumps(condition), '--expect-count', '0'], 0)
        check(['--metadata-condition', '{"conditions":[],"conditions":[]}'], 2)
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=5) == 0
        print('CHECKER DEMO PASS: installed CLI; filtered count/title assertions; explicit zero hits; mismatches exit 1; duplicate JSON exits 2; reports redacted')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


if __name__ == '__main__':
    main()
