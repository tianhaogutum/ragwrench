import sys

from ragwrench.cli import main

raise SystemExit(main(["highlight", *sys.argv[1:]]))
