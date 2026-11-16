import sys

from ragwrench.cli import main

raise SystemExit(main(["cite-check", *sys.argv[1:]]))
