"""The MCP surface: an agent queries the index, and supplies the inference itself.

This server inverts the usual arrangement. A retrieval tool that wanted to answer
"what does this function guarantee" would have to call a model, which means an API
key, a bill, a rate limit, and a second place where an unciteable sentence can be
born. Here the agent is already running and already paid for, so it calls *in*:
the tools do the deterministic half -- parse, retrieve, hash, gate, store -- and the
agent supplies the judgement through `submit_assertion`.

That is only worth anything because the gate is not a matter of opinion. It checks
that every cited span exists at the lines given and that its bytes still hash to
what was cited. Both are arithmetic. An agent cannot talk its way past a sha256, and
when it fails it is told exactly which citation moved and what the file says now --
so the next attempt is a correction rather than another guess.

Everything else here is a thin projection of code that already exists and is already
tested: `retrieve.search`, `onboard.build_reading_path`, `assertions.store`, and the
tier derivation in `codelearner.tier`. Deliberately thin. Two surfaces over one
index that each derive "which tier is this hit" separately will disagree eventually,
and nothing will say which one is lying.

That derivation used to be `cli.render`'s, which meant this package reached upward
into the one a person types in order to answer a machine. Sharing it was right;
letting one of the two surfaces own it was not.
"""

from .app import SERVER_NAME, ToolError, build_server
from .main import main

__all__ = ["SERVER_NAME", "ToolError", "build_server", "main"]
