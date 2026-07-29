"""`python -m codelearner.cli` -- the same entry point without an installed script.

Worth having: the console script only exists after `pip install`, and the first
thing anyone does with a checkout is run it out of the source tree.
"""
import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
