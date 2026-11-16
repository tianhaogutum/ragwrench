"""Shared configuration for all tool suites.

Suites whose optional dependencies are not installed are skipped at collection, so
``pip install -e .[test]`` runs the core tools and ``.[all,test]`` runs everything.
"""
import importlib.util


def _missing(*modules: str) -> bool:
    return any(importlib.util.find_spec(name) is None for name in modules)


collect_ignore = []
if _missing("qdrant_client"):
    collect_ignore += ["sync", "bm25", "difybridge"]
if _missing("chromadb", "qdrant_client"):
    collect_ignore += ["migrate"]
