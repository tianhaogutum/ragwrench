import sys

from ragwrench.cli import main

raise SystemExit(main(["port", *sys.argv[1:]]))
