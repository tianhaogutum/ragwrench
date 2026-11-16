import sys

from ragwrench.cli import main

raise SystemExit(main(["eval", *sys.argv[1:]]))
