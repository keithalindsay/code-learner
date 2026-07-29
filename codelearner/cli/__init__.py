"""The human surface: `codelearner index`, `search`, and `stats`.

Everything below this package is an API. This is the part a person types.
"""

from .main import build_parser, main

__all__ = ["build_parser", "main"]
