import sys

from ragwrench.cli import main

raise SystemExit(main(["audit", *sys.argv[1:]]))
