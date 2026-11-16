import sys

from ragwrench.cli import main

raise SystemExit(main(["migrate", *sys.argv[1:]]))
