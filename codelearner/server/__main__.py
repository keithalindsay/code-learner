"""`python -m codelearner.server` -- the same entry point without an installed script.

Worth having twice over here. The console script only exists after `pip install`,
and an MCP client's config block names a `command` that has to resolve to something
executable on someone else's machine -- so pointing it at `<venv>/bin/python -m
codelearner.server` is the configuration that works from a bare checkout, before
anything has been installed at all.
"""
import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
