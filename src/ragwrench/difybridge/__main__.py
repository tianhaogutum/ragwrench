import sys

from ragwrench.cli import main

raise SystemExit(main(["dify-bridge", *sys.argv[1:]]))
