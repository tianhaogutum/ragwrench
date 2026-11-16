"""Incremental, inspectable indexing for local Qdrant."""
__version__ = "0.1.0"

# Listed by ``ragwrench sync --help`` when the qdrant extra is missing.
SUBCOMMANDS = {
    "snapshot": "snapshot UTF-8 documents or already split JSONL chunks",
    "plan": "read-only diff; exit 10 if snapshots differ",
    "compare": "compare two plans from the same source; exit 10 if different",
    "compare-policies": "compare archived inventories without store access",
    "review": "evaluate plan gates; exit 3 if blocked, 2 for invalid input",
    "policy-override": "record a separate plan/collection-bound policy weakening exception",
    "inspect-policy": "read a local collection's effective policy and latest exception",
    "list-policies": "inventory local catalog policies without opening points",
    "apply": "apply a plan to a local Qdrant collection",
    "verify": "verify a local Qdrant collection against a snapshot",
    "bench": "replay synthetic edit scenarios and write measured counts",
}
