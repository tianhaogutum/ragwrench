"""Reproduce Linux CLI peak-RSS/content parity on identical synthetic Markdown.

Run from an installed environment: python benchmarks/streaming.py -o results.json
Each mode runs in a fresh process. RSS includes Python/imports and all CLI stages;
output SHA256 covers every JSONL byte, including provenance. No speed assertion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            value.update(block)
    return value.hexdigest()


def worker(source, output, streaming):
    from ragwrench.tablechunk.cli import main
    args = ['split', source, '--max-tokens', '256', '-o', output]
    if streaming:
        args.append('--stream')
    start = time.perf_counter()
    result = main(args)
    elapsed = time.perf_counter() - start
    if result:
        raise SystemExit(result)
    print(json.dumps({'elapsedSeconds': elapsed,
                      'peakRssKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))


def measure(rows):
    with tempfile.TemporaryDirectory(prefix='tablechunk-parity-') as directory:
        source = Path(directory) / 'table.md'
        with source.open('w', encoding='utf-8') as stream:
            stream.write('# Catalog\n\nAnnual rates.\n\n| ID | Seats | Price | Notes |\n|---|---|---|---|\n')
            for index in range(rows):
                stream.write(f'| R{index:07d} | {index % 100 + 1} | ${index * 5} | annual subscription region west |\n')
        measurements = {}
        hashes = []
        for mode in ('buffered', 'stream'):
            output = Path(directory) / f'{mode}.jsonl'
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                                  str(source), str(output), mode], check=True, capture_output=True, text=True)
            result = json.loads(run.stdout)
            result['outputSha256'] = digest(output)
            with output.open('rb') as stream:
                result['chunks'] = sum(1 for _ in stream)
            result['outputBytes'] = output.stat().st_size
            hashes.append(result['outputSha256'])
            measurements[mode] = result
        assert hashes[0] == hashes[1], 'JSONL parity failure'
        return {'rows': rows, 'inputBytes': source.stat().st_size, 'inputSha256': digest(source),
                'identicalJsonl': True, 'modes': measurements}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3], sys.argv[4] == 'stream')
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', type=int, default=[5000, 50000])
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('RSS units in this experiment require Linux')
    result = {'schemaVersion': 1, 'python': sys.version, 'platform': sys.platform,
              'maxTokens': 256, 'mode': 'markdown', 'adapter': 'jsonl',
              'method': 'fresh subprocess per CLI mode; Linux ru_maxrss KiB; wall seconds; SHA256 of full JSONL',
              'cases': [measure(rows) for rows in args.rows]}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
