import sys

from ragwrench.cli import main

raise SystemExit(main(["sync", *sys.argv[1:]]))
