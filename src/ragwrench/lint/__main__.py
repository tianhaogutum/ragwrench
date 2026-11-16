import sys

from ragwrench.cli import main

raise SystemExit(main(["lint", *sys.argv[1:]]))
