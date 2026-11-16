"""Regenerate original synthetic format fixtures, without importing an upstream.

Run from any directory: python examples/lint/generate.py
The Markdown document is the input; all writes stay beside this script or in the
explicit package data directory. No production or downloaded content is used.
"""
import json
from pathlib import Path


def generate():
    directory = Path(__file__).resolve().parent
    source = (directory / 'doc.md').read_text(encoding='utf-8')
    boundaries = [0, source.index('## Code'), source.index('print("a shorter'),
                  source.index('| beta'), source.index('## Usage'), source.index('Run the local'), len(source)]
    parts = [source[a:b] for a, b in zip(boundaries, boundaries[1:])]
    records = [{'id': f'lc{i + 1}', 'page_content': text,
                'metadata': {'source': 'synthetic://guide', 'producer': 'langchain-document-shape'}}
               for i, text in enumerate(parts)]
    records.append({**records[-1], 'id': 'duplicate'})
    (directory / 'chunks_langchain.jsonl').write_text(
        ''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
    nodes = [{'id_': record['id'], 'text': record['page_content'], 'metadata': record['metadata'],
              'start_char_idx': source.index(record['page_content']),
              'end_char_idx': source.index(record['page_content']) + len(record['page_content'])}
             for record in records]
    (directory / 'chunks_llamaindex.json').write_text(json.dumps(nodes, indent=2) + '\n', encoding='utf-8')
    package = directory.parents[1] / 'src/ragwrench/lint/data'
    for name in ('doc.md', 'chunks_langchain.jsonl'):
        (package / name).write_bytes((directory / name).read_bytes())


if __name__ == '__main__':
    generate()
