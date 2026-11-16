"""``ragwrench audit``: one read-only health report for a chunk export.

The checks are the ``ragwrench lint`` rules; the repair preview is ``ragwrench lint fix``
run in memory. Nothing is written.
"""
from .. import __version__

SCHEMA = "ragwrench.audit/v1"

__all__ = ["SCHEMA", "__version__"]
