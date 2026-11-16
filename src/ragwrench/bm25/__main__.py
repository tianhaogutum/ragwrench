import sys

from ragwrench.cli import main

raise SystemExit(main(["bm25", *sys.argv[1:]]))
