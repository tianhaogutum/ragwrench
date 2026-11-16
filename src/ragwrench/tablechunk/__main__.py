import sys

from ragwrench.cli import main

raise SystemExit(main(["table-chunk", *sys.argv[1:]]))
