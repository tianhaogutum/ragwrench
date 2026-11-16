"""Example JSONL command protocol; deterministic hash vectors, not ML inference."""
import argparse
import json
import sys

from ragwrench.sync.embedders import HashEmbedder

parser = argparse.ArgumentParser()
parser.add_argument("--dimension", type=int, default=8)
args = parser.parse_args()
embedder = HashEmbedder(args.dimension)
for line in sys.stdin:
    row = json.loads(line)
    vector = embedder.embed(row, row["embed_model"])
    print(json.dumps({"id": row["id"], "vector": vector}), flush=True)
