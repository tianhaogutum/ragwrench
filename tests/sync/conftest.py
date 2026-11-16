"""Platform gates for the sync suite."""
import sys

import pytest

# Persistent sync stores write the recovery sidecar with file and directory
# fsync, which ragwrench supports only on local Linux filesystems (docs/sync.md);
# elsewhere every persistent apply refuses up front. Tests that need such writes
# are skipped off Linux; pure planning/policy/in-memory tests still run.
DURABLE_REASON = "sync durable recovery writes are Linux-only (local filesystem with file/directory fsync)"
linux_durable = pytest.mark.skipif(sys.platform != "linux", reason=DURABLE_REASON)
