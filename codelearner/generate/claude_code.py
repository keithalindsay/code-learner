"""The same two seams as `llm.py`, behind the `claude` CLI instead of ollama.

`llm.py` writes claims with `llama3.1:8b`, and the corrected instruments say that
generator is significantly *worse* than a bag of body identifiers. This module exists to
ask whether that is a fact about the method or a fact about an 8B model, and it is built
so that the answer to that question is the only thing that changes between the two runs.

**The prompt is not re-written here, it is re-used.** `build_generation_prompt` and
`purpose.build_prompt` are imported and sent verbatim -- same system text, same numbered
menu, same rules, same bad/good pair. A prompt tuned for Claude would produce a number
that answers a different question than the one asked, and there would be no way to tell
from the scorecard which question it answered. The only intended difference between an
ollama row and a claude-code row is the weights.

## How the (system, user) split survives a CLI that takes one prompt

`claude -p` takes a single positional prompt, so the split has to be carried by a flag.
There are two candidates and only one of them is honest:

- `--append-system-prompt` keeps Claude Code's own agent system prompt -- tool
  instructions, working directory, environment blurb, git status -- and appends ours
  underneath. That is not the instruction `llama3.1:8b` was given, and the difference is
  thousands of tokens of unrelated framing.
- `--system-prompt` REPLACES the default system prompt with exactly the string given.

So `--system-prompt` carries `_SYSTEM_PROMPT` byte for byte and the user message is the
positional argument, byte for byte. Nothing is concatenated and nothing is summarised;
`build_argv` is a pure function of the pair, and a test asserts that what lands in argv
is what `build_generation_prompt` returned rather than an f-string that resembles it.

The residual difference, stated rather than hidden: the ollama request carries no tool
definitions and an API request from Claude Code may still carry some in the `tools`
parameter (which is not system text and cannot be removed from the prompt side). Passing
`--tools ""` removes them at the source, which is done here for a stronger reason -- see
below -- and has this as a welcome side effect.

## Tool denial is a correctness requirement, and the obvious way to do it does not work

The purpose eval's `docstring_blind` condition strips the docstring out of the source and
hands the model a `SourceView` with no path. A subprocess that can open files could
re-read the file and recover exactly what was stripped, or read `git log` and find the
held-out gold label itself. The eval would then report a strong doc-blind result that
means nothing at all, and a *better* score is the one outcome nobody investigates.

The obvious mechanism is `--disallowedTools Read Write Edit Bash ...`. **It was measured
and it does not hold.** Planting a file with a distinctive token in the working directory
and asking `claude -p` to read it, with that flag set and with `--safe-mode` and
`--strict-mcp-config` on top, returned the token. The stream shows what happened:

    ToolSearch {"query": "select:Read"}          -> No matching deferred tools found
    ToolSearch {"query": "read file contents bash shell command"}
                                                 -> [Monitor, TaskOutput, ...]
    Monitor    {"command": "cat canary.txt"}     -> the token

`--disallowedTools` is a blocklist over a tool namespace that is open, versioned, and
searchable -- and the model searched it. `Monitor` runs a shell command and is not named
`Bash`, so it was never denied, and `permission_denials` came back EMPTY: from the
harness's point of view nothing was refused, because nothing that was refused was tried.
A blocklist against a namespace the other party can enumerate is not a boundary.

`--tools ""` is the allowlist form: it selects the set of available built-in tools, and
the empty set means the model is handed no tools at all. The same probe under it returns
`num_turns: 1` and no tool call -- there is nothing to search for and nothing to deny.
That is the mechanism this module relies on, and `--disallowedTools` is kept beside it as
a second layer and as a record in argv of what was intended.

Three more flags, each closing a channel the blocklist would have left open:

- `--strict-mcp-config` with no `--mcp-config`: MCP tools are named `mcp__server__tool`
  and cannot be enumerated in advance, so a machine with a browser-automation or
  shell-adjacent MCP server configured would hand the subprocess a file reader under a
  name no deny list could have predicted.
- `--setting-sources ""`: the user, project and local settings files are not loaded. On
  the machine this was built on, `settings.local.json` contains `Read(//tmp/**)` in its
  allow list. A measurement whose leak boundary depends on what is in the operator's
  personal settings file is not a boundary, and this also removes hooks, which can run
  arbitrary commands on tool events.
- `--safe-mode`: no `CLAUDE.md`, no skills, no plugins, no custom agents. Those are
  instructions the ollama run did not receive, so they are a prompt confound before they
  are a leak risk.

**The denial is checked live, not asserted.** Passing a flag and having a flag work are
different claims, and this module was written because they came apart. `tests/
test_claude_code.py` carries an opt-in live test that plants a token, runs a permissive
control to prove the plant is actually reachable, then runs the shipped configuration and
asserts the token is absent. Without the control the test would pass just as green if the
plant had silently failed.

## The name is the model that answered, not the model that was asked for

`name` lands in `assertions.generator` and in scorecard rows, and it is the only thing
that lets a store holding claims from two generators tell them apart. Taking it from the
`--model` argument would be a provenance bug: the harness can answer with something else,
and a row labelled with a request rather than with a fact is worse than an unlabelled row
because it looks trustworthy. So it is read out of `modelUsage` in the response envelope.

Two consequences. The name cannot be known before something has been asked, so the first
read of `name` resolves it with a minimal probe call (the pipeline reads `name` once,
before the loop, to look up which subjects are already claimed). And every subsequent call
re-checks it: a mid-run fallback to another model would put two models' claims under one
`generator` string, which is precisely the confusion that column exists to prevent, so it
raises `ModelSubstituted` and stops the run rather than mislabelling the remainder.

## What the envelope actually does, including one shape that lies

Measured on v2.1.220, `--output-format json`:

- Success: `type: "result"`, `subtype: "success"`, `is_error: false`, `result` holding the
  model's raw text, `modelUsage` keyed by model id, `permission_denials`, `num_turns`,
  `total_cost_usd`, `duration_ms`, `stop_reason`, `terminal_reason`, `api_error_status`.
- **Failure also says `subtype: "success"`.** A request for a model that does not exist
  exits 1 and returns `type: "result"`, `subtype: "success"`, `is_error: true`,
  `modelUsage: {}`, `terminal_reason: "api_error"`, `api_error_status: 404`, and a
  `result` holding a human-readable error sentence: *"There's an issue with the selected
  model (...)"*.

That last shape is why `subtype` is treated here as a reliable failure signal and never as
a success signal. A backend that trusted it would have parsed an English error message as
the model's answer, found no JSON in it, and recorded an empty draft -- writing "these
spans establish nothing" against every symbol in the repo because the CLI could not reach
a model. That is exactly the outage-mistaken-for-refusal collapse `GeneratorUnavailable`
exists to prevent, arriving through a door `is_error` alone would have closed.

## Attribution is a membership test, because counting tokens broke a real run

`modelUsage` routinely holds more than one entry: a `claude-haiku-4-5` entry appears
beside the answering model, having spent a few tokens on harness-internal work. Something
has to decide which of them the claim belongs to.

The first version of this module took the entry with the most OUTPUT tokens, on the theory
that the answer is the thing that was written. **It broke a 151-symbol run at symbol 105
and made 46 symbols unreachable**, and the way it broke is the point:

    result : '{"claim": "", "cited_refs": []}'
    claude-haiku-4-5-20251001   output_tokens=17
    claude-opus-5[1m]           output_tokens=18

That is a correct abstention -- a first-class answer in this design, which `llm.py` states
in bold -- and it is about eighteen tokens long, one token clear of the harness's
incidental usage. Argmax therefore sat on a knife edge that tipped whenever the generator
did the *best* thing available to it, filed a legitimate abstention as a fallback to
another model, and aborted the run. The guard fired hardest exactly where the generator
was behaving well, which is the worst possible place for a guard to fire.

The rule is now a membership test, and the question it asks is **"did the model this run
is named for participate at all?"** An entry exists for a model whenever that model ran,
no matter how little it wrote, so:

- the run's model present in `modelUsage` -> it answered; any other entry is overhead;
- the run's model **absent** -> that, and only that, is a substitution.

Nothing is counted, so nothing can be one token away from the wrong answer. The failure
direction inverts: a quiet answer can no longer look like a substitution, because absence
is not something brevity can produce, and overhead can never be mistaken for the run's
model because it carries a different id.

**Resolving the name the first time** is the one place a choice still has to be made, and
it is made by refusing to guess. One entry, and there is nothing to choose -- `_run` has
already refused any envelope without a textual `result`, so the session model ran and has
an entry, and a lone entry beside a real answer names it rather than winning a comparison.
More than one, with no `model=` pinned, and the call raises rather than picking a
favourite; the message names the candidates and the argument that settles it.

That is why the probe is two words long. Harness side-work scales with the request: the
same probe carrying the real ~850-token system prompt comes back with a `claude-haiku-4-5`
entry alongside the answering model, while the two-word version comes back with one entry.
Probing with the real prompt looks more rigorous and manufactures the exact ambiguity the
resolution must not have.

**`--model` is NOT passed by default, and the reason is measured.** Pinning removes the
attribution question entirely -- the request is matched by identity from the first call
and a harness fallback is caught before a single claim is written -- but `--model opus`
reports itself as `claude-opus-5` while the unpinned call on the same account reports
`claude-opus-5[1m]`. Those are the same weights and two different strings in
`assertions.generator`, so switching the default would orphan every claim an earlier run
had already stored and split the column that exists to keep runs comparable. The `[1m]`
suffix is also an entitlement of one account and cannot be a library default. So the
default stays unpinned, `model=` is available for a caller who wants identity matching from
call one, and the trade is written down here rather than discovered later.

## Fail closed, and keep outage apart from refusal

The boundary is the same as `llm.py`'s and is not weaker for being a subprocess. A CLI
that is missing, a non-zero exit, a timeout, `is_error: true`, stdout that is not JSON,
JSON that is not an object, a missing or non-string `result`, an empty `modelUsage` --
none of these is a model declining to make a claim, and all of them raise
`GeneratorUnavailable`. A model that answered with something `parse_draft` cannot read is
a refusal and returns an empty `Draft`, which the pipeline counts and refuses.

`parse_draft` is imported rather than re-implemented: the response schema is the same
schema, and a second parser would be a second place for the citation rules to drift.

## Cost, and why there is no `release()`

Its siblings hold VRAM and must be unloaded before the judge can fit on the card. This one
holds nothing locally; what it spends is a Max plan's usage allowance, at roughly 8-25
seconds per call. `release()` is therefore absent rather than a no-op, so a caller cannot
believe it is managing a resource that does not exist.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess  # noqa: S404 - fixed argv, no shell; every failure raises GeneratorUnavailable
from collections.abc import Mapping, Sequence
from pathlib import Path

from .llm import DEFAULT_KIND, build_generation_prompt, collides_with_judge, parse_draft
from .types import Draft, GeneratorUnavailable, Offer

logger = logging.getLogger(__name__)

__all__ = [
    "CLAUDE_LINEAGE",
    "DEFAULT_CLI_NAME",
    "DEFAULT_TIMEOUT_S",
    "DENIED_TOOLS",
    "FALLBACK_CLI_PATH",
    "NAME_PREFIX",
    "STRIPPED_ENV_VARS",
    "ClaudeCodeClaimGenerator",
    "ClaudeCodePurposeModel",
    "ModelSubstituted",
    "answering_model",
    "build_argv",
    "resolve_cli",
]


# The command, looked up on PATH. Not an absolute path constant: the CLI is installed per
# user and the machine this was written on does not have it on PATH at all, so a constant
# would be one person's home directory baked into a measurement harness.
DEFAULT_CLI_NAME = "claude"

# Where the local installer puts it when PATH does not have it. Tried second, expanded at
# call time so it follows `$HOME` rather than the author's.
FALLBACK_CLI_PATH = "~/.claude/local/claude"

# `assertions.generator` gets `claude-code/<model id>`. Prefixed with the RUNTIME for the
# same reason `ollama/` is: the same weights behind a different harness, system prompt or
# tool configuration are not guaranteed to be the same generator, and this column is the
# only thing that survives to make a before/after comparison possible.
NAME_PREFIX = "claude-code"

# The lineage word every model this backend can reach belongs to. Used to ask the
# collision question BEFORE any call has resolved a real model id -- the hazard is a
# property of the vendor, not of which Claude answered. See `collides_with_judge`.
CLAUDE_LINEAGE = "claude"

# 25s was the slowest observed call with tools enabled and 8s the fastest without. The
# ceiling is generous for the same reason as `llm.DEFAULT_TIMEOUT_S`: a timeout that fires
# mid-generation costs the whole call and returns nothing, which is worse than waiting.
DEFAULT_TIMEOUT_S = 300.0

# The blocklist, kept as a SECOND layer under `--tools ""` and as a record in argv.
#
# It is not the boundary and must not be read as one -- the module docstring records the
# run where a model got past this exact list by asking `ToolSearch` for something that
# reads files and being handed `Monitor`, which runs a shell command and is not called
# `Bash`. `ToolSearch` and `Monitor` are named here because they are the measured escape
# route and a list that did not name them would look complete.
DENIED_TOOLS = (
    "Read",
    "Write",
    "Edit",
    "NotebookEdit",
    "Bash",
    "BashOutput",
    "KillShell",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "Agent",
    "Task",
    "TaskOutput",
    "ToolSearch",
    "Monitor",
    "Skill",
    "SlashCommand",
)

# Environment variables removed from the child's environment.
#
# Every one of them is set by Claude Code itself, so they are present exactly when the
# harness is driven from inside a Claude Code session and absent when it is run from a
# plain shell. A measurement that changes depending on how the operator happened to launch
# it is not a measurement, and `CLAUDE_EFFORT` is the sharp end of that: inherited, it
# would set the child's reasoning effort from the parent session's setting, which is a
# large uncontrolled variable sitting directly on the thing being measured.
#
# Authentication variables are deliberately NOT stripped. `ANTHROPIC_API_KEY` and its
# relatives are how a non-interactive run authenticates, and removing them would turn a
# configuration difference into an outage.
STRIPPED_ENV_VARS = (
    "CLAUDE_EFFORT",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_BRIDGE_SESSION_ID",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_PID",
    "CLAUDECODE",
    "AI_AGENT",
)

# The envelope's `result` is the model's raw text; these are the fields consulted to
# decide whether it is an answer at all.
_ENVELOPE_TYPE = "result"
_ENVELOPE_SUBTYPE = "success"


class ModelSubstituted(GeneratorUnavailable):
    """The harness answered with a different model than the one already recorded.

    A subclass of `GeneratorUnavailable` rather than a new seam, and that is the whole
    design: `pipeline.learn` lets `GeneratorUnavailable` out uncaught and stops the run,
    which is the correct response here. The model answered, so this is not literally an
    outage -- but the answer cannot be stored, because `name` was read once before the
    loop and every row from here on would be filed under a model that did not write it.
    Half a run of correctly-labelled claims is recoverable. A run of claims labelled with
    the wrong model is not, and nothing downstream can detect it.
    """


def resolve_cli(cli: str | None = None) -> str:
    """The `claude` executable to run: an explicit path, then PATH, then the local install.

    Returns a string rather than raising when nothing is found, so a missing CLI surfaces
    at the call as `GeneratorUnavailable` with the argv in the message, alongside every
    other reason the backend did not answer. Raising from a constructor would put the one
    failure an operator can fix in a different place from all the others.
    """
    if cli:
        return cli
    found = shutil.which(DEFAULT_CLI_NAME)
    if found:
        return found
    return str(Path(FALLBACK_CLI_PATH).expanduser())


def build_argv(
    *,
    cli: str,
    system: str,
    user: str,
    model: str | None = None,
) -> list[str]:
    """The full argv for one non-interactive call. Pure, and exposed so it can be diffed.

    Named and public for the same reason `build_generation_prompt` is: the invocation is
    as much of the experimental condition as the prompt, and two runs whose numbers differ
    should be able to establish whether the sandbox was the same. It is also what the unit
    tests assert against, since argv is the only place the denials are visible without a
    live call.

    The order of the two prompt halves is the point. `--system-prompt` carries the system
    text unchanged -- it REPLACES Claude Code's default agent prompt rather than appending
    to it, which is what keeps the instruction identical to the one ollama was sent -- and
    the user text is the positional argument, also unchanged. Nothing is concatenated.

    `--model` is passed only when a caller names one. Left off, the CLI picks; either way
    the name written against the claim comes from `modelUsage` in the response, never from
    this argument.

    The menu is bounded by `pipeline.DEFAULT_MAX_OFFERS` (12) times
    `DEFAULT_MAX_OFFER_BYTES` (4000), so the largest user prompt is ~48KB against a Linux
    `ARG_MAX` of ~2MB. Passing it in argv is safe at that size and keeps this function
    pure; a caller that raises those caps by a factor of forty gets an `OSError` from
    `subprocess.run`, which is reported as an outage rather than as a refusal.
    """
    argv = [cli, "-p", "--output-format", "json"]
    if model:
        argv += ["--model", model]
    argv += [
        # The boundary. An allowlist of built-in tools, empty: the model is handed no
        # tools, so there is no namespace to search and nothing to deny. See the module
        # docstring for the measured run where the blocklist below was not enough.
        "--tools",
        "",
        # Second layer, and the record of intent. Harmless when `--tools ""` holds and the
        # only thing standing between the eval and the file system if a future CLI changes
        # what an empty tool list means.
        "--disallowedTools",
        *DENIED_TOOLS,
        # No MCP servers. Their tools are named `mcp__server__tool` and cannot be
        # enumerated in advance, so this is the only way to deny them.
        "--strict-mcp-config",
        # No user, project or local settings: no allow rules that could out-rank the deny
        # list, and no hooks, which run commands of their own on tool events.
        "--setting-sources",
        "",
        # No CLAUDE.md, skills, plugins or custom agents. Instructions the ollama run never
        # received are a prompt confound before they are a leak.
        "--safe-mode",
        "--system-prompt",
        system,
        user,
    ]
    return argv


def child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment the subprocess gets: this one, minus the session-context variables.

    See `STRIPPED_ENV_VARS`. The property being bought is that a call made from inside a
    Claude Code session and the same call made from a bare shell produce the same request.
    """
    env = dict(os.environ if base is None else base)
    for key in STRIPPED_ENV_VARS:
        env.pop(key, None)
    return env


# A trailing context-window entitlement (`claude-opus-5[1m]`) and a trailing release date
# (`claude-haiku-4-5-20251001`) are decoration on the same identity, and both appear and
# disappear depending on how the model was asked for. They are stripped before any
# comparison so that `--model claude-opus-5` matching `claude-opus-5[1m]` is an identity
# and not a near-miss to be guessed at.
_ID_DECORATION = re.compile(r"\[[^]]*\]\s*$|-\d{8}$")


def _identity(model_id: str) -> str:
    """A model id reduced to the part that names the weights. Comparison happens here."""
    return _ID_DECORATION.sub("", model_id.strip().lower()).strip()


def _entry_identities(envelope: Mapping[str, object]) -> dict[str, set[str]]:
    """Every `modelUsage` id, mapped to the identities it answers to.

    `canonicalModel` is carried in the envelope beside each entry and is the harness's own
    answer to "which model is this really" -- `claude-opus-5[1m]` canonicalises to
    `claude-opus-5`. It is read rather than re-derived, because a rule this module invents
    for collapsing ids would be a second opinion about a fact the harness already states.
    """
    usage = envelope.get("modelUsage")
    if not isinstance(usage, Mapping) or not usage:
        raise GeneratorUnavailable(
            "the claude CLI returned an envelope with no `modelUsage`, so the model that "
            "answered cannot be named. A claim filed under a model that did not write it "
            "is worse than a lost call, and an empty `modelUsage` is also what a failed "
            "call returns -- so this is treated as no answer at all."
        )
    entries: dict[str, set[str]] = {}
    for model_id, stats in usage.items():
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        names = {_identity(model_id)}
        if isinstance(stats, Mapping):
            canonical = stats.get("canonicalModel")
            if isinstance(canonical, str) and canonical.strip():
                names.add(_identity(canonical))
        entries[model_id] = names
    if not entries:
        raise GeneratorUnavailable(
            f"the claude CLI returned a `modelUsage` with no usable model id: "
            f"{envelope.get('modelUsage')!r}"
        )
    return entries


def _answers_to_request(names: set[str], request: str) -> bool:
    """Whether an entry answers to what a caller asked `--model` for.

    Looser than the identity test above, because `--model` accepts an alias: `opus` is a
    legal way to ask for `claude-opus-5`. So a request also matches when it is one of the
    hyphen-delimited words of an identity. That is deliberately not symmetric with the
    strict comparison used once a run has been named -- an alias is how a request is
    written, never how an answer is recorded, and `opus` matching `claude-opus-5` must not
    also make `claude-opus-5` acceptable where `claude-opus-6` was recorded.
    """
    wanted = _identity(request)
    return any(wanted == name or wanted in name.split("-") for name in names)


def answering_model(
    envelope: Mapping[str, object],
    *,
    expected: str | None = None,
    requested: str | None = None,
) -> str:
    """Which model in `modelUsage` this claim belongs to. A membership test, never a count.

    Raises `GeneratorUnavailable` when `modelUsage` is missing or empty. That is a strong
    reaction to a bookkeeping field and it is deliberate: the id is what gets written into
    `assertions.generator`, an unlabelled claim is not something this project has a use
    for, and an empty `modelUsage` is also the observed shape of an envelope whose call
    failed. Losing the call is cheap; a store of claims nobody can attribute is not.

    Three cases, in the order they are asked:

    **`expected` -- the run already has a name.** The only question is whether that model
    participated, and an entry exists for a model whenever it ran, however little it wrote.
    Present: attribute to it, and keep the ORIGINALLY recorded string as the name even if
    the decoration moved, so the `generator` column does not split mid-run. Absent: that is
    a substitution, and `ModelSubstituted` stops the run.

    **`requested` -- a caller pinned `--model`.** Identity matching against the request
    from the very first call, so a harness fallback is caught before any claim is written.

    **Neither.** One entry and there is nothing to choose. More than one and this REFUSES,
    naming the candidates: guessing is what broke a run before (see the module docstring),
    and an operator adding `model=` is a ten-second fix while a store attributed to the
    wrong model is not fixable at all.

    Nothing here counts tokens. That is the whole correction: the previous rule took the
    entry with the most output, and a correct abstention is about eighteen tokens against
    an incidental seventeen, so the guard misfired precisely when the generator was doing
    the best thing available to it.
    """
    entries = _entry_identities(envelope)

    if expected is not None:
        wanted = _identity(expected)
        for model_id, names in entries.items():
            if wanted in names:
                if model_id != expected:
                    logger.warning(
                        "the run is recorded as %r and this response reports the same "
                        "model as %r. Same weights, different decoration; the recorded "
                        "name is kept so the generator column does not split mid-run.",
                        expected,
                        model_id,
                    )
                return expected
        raise ModelSubstituted(
            f"this run is recorded as {NAME_PREFIX}/{expected} and that model does not "
            f"appear in the response at all -- it reports {sorted(entries)}. Continuing "
            "would file another model's claims under this run's generator name, which is "
            "the confusion that column exists to prevent. Re-run with an explicit "
            "`model=` once the fallback has cleared."
        )

    if requested:
        matched = [m for m, names in entries.items() if _answers_to_request(names, requested)]
        if len(matched) == 1:
            return matched[0]
        if not matched:
            raise ModelSubstituted(
                f"{requested!r} was requested and no model answering to that name appears "
                f"in the response, which reports {sorted(entries)}. The harness answered "
                "with something else, so nothing from this call can be attributed."
            )
        raise GeneratorUnavailable(
            f"{requested!r} matches {sorted(matched)} in the same response, so this claim "
            "cannot be attributed to one model. Pass a fully qualified `model=` id."
        )

    if len(entries) == 1:
        return next(iter(entries))
    raise GeneratorUnavailable(
        f"the response reports {sorted(entries)} and no model was pinned, so which one "
        "this run should be named for cannot be established. Guessing is what this rule "
        "exists to stop -- pass `model=` to name it."
    )


def _run(argv: Sequence[str], *, timeout: float, role: str, outage_detail: str) -> dict[str, object]:
    """One CLI invocation, raising `GeneratorUnavailable` for anything that is not an answer.

    The subprocess analogue of `llm._chat_post`, and it carries the same boundary: a
    backend that did not answer must never be turned into a `Draft`, because an empty
    draft is a claim about the code ("these spans establish nothing") and an outage is a
    claim about the machine. `role` and `outage_detail` let each caller say what was lost
    in its own terms, exactly as its ollama sibling does.

    Every check here is a measured failure shape rather than a precaution. In particular
    `is_error` is consulted and `subtype` is not trusted to say the opposite: an envelope
    from a call that failed at the API comes back with `subtype: "success"` AND
    `is_error: true`, so a reader that keyed off `subtype` would parse the human-readable
    error sentence in `result` as the model's answer.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user-built flags
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=child_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise GeneratorUnavailable(
            f"the {role} did not answer within {timeout:g}s. {outage_detail}"
        ) from exc
    except OSError as exc:
        raise GeneratorUnavailable(
            f"could not run the {role} ({argv[0]!r}): {exc}. {outage_detail} "
            "Install the claude CLI, or pass its path as `cli=`."
        ) from exc

    if completed.returncode != 0:
        raise GeneratorUnavailable(
            f"the {role} exited {completed.returncode}. {outage_detail} "
            f"stderr: {completed.stderr.strip()[:400]!r}"
        )
    try:
        envelope = json.loads(completed.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        raise GeneratorUnavailable(
            f"the {role} wrote stdout that is not JSON ({exc}). {outage_detail} "
            f"stdout: {completed.stdout.strip()[:400]!r}"
        ) from exc
    if not isinstance(envelope, dict):
        raise GeneratorUnavailable(
            f"the {role} returned {type(envelope).__name__}, not an envelope object."
        )

    # `is_error` first, because it is the field that is true when the call failed. A
    # failing envelope also reports `subtype: "success"`, so the subtype check below can
    # only ever be a second opinion about failure and is never evidence of success.
    if envelope.get("is_error"):
        raise GeneratorUnavailable(
            f"the {role} reported is_error with terminal_reason "
            f"{envelope.get('terminal_reason')!r} and api_error_status "
            f"{envelope.get('api_error_status')!r}. {outage_detail} "
            f"It said: {str(envelope.get('result'))[:300]!r}"
        )
    if envelope.get("type") != _ENVELOPE_TYPE:
        raise GeneratorUnavailable(
            f"the {role} returned type {envelope.get('type')!r}, not {_ENVELOPE_TYPE!r}. "
            f"{outage_detail}"
        )
    subtype = envelope.get("subtype")
    if subtype is not None and subtype != _ENVELOPE_SUBTYPE:
        raise GeneratorUnavailable(
            f"the {role} returned subtype {subtype!r} (for example a turn limit). "
            f"{outage_detail}"
        )
    if not isinstance(envelope.get("result"), str):
        raise GeneratorUnavailable(
            f"the {role} returned an envelope with no textual `result`. {outage_detail}"
        )

    denials = envelope.get("permission_denials")
    if isinstance(denials, list) and denials:
        # Not an error: a denial is the sandbox working. It is warned about because under
        # `--tools ""` the model should have had nothing to attempt, so a denial means
        # tools reached it after all and the boundary is resting on the blocklist -- which
        # is measured NOT to hold on its own.
        logger.warning(
            "the %s attempted %d tool call(s) and was denied: %r. The subprocess was "
            "meant to have no tools at all; the leak boundary is now resting on the "
            "deny list, which is known to be escapable.",
            role,
            len(denials),
            denials[:5],
        )
    return envelope


class _ClaudeCodeBackend:
    """Shared transport, naming and model-substitution check for the two seams below.

    Extracted for the reason `llm._chat_post` was: the duplication this package keeps on
    purpose is the duplication ACROSS packages, so that the thing being measured cannot
    import the thing measuring it. That argument says nothing about two classes in one
    file, where a second copy would only be a second place for the outage semantics and
    the provenance rule to drift apart.
    """

    _role = "claude generator"
    _outage_detail = "No answer was produced."

    def __init__(
        self,
        model: str | None = None,
        *,
        cli: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._requested_model = model
        self._cli = resolve_cli(cli)
        self._timeout = timeout
        self._answered_model: str | None = None
        # Ids seen beside the run's own model. Harness overhead is expected and is not an
        # error, but "the extra entries are overhead" is an assumption, and an assumption
        # that is never checked is how the previous rule survived long enough to break a
        # run. A model appearing here that was not there before is worth one warning.
        self._auxiliary_models: set[str] = set()
        if collides_with_judge(CLAUDE_LINEAGE):
            logger.warning(
                "the faithfulness judge is in the %r lineage, which is this generator's. "
                "The faithfulness score assumes the judge did not write the claim; with "
                "this pairing it stops being an audit and becomes self-assessment, and "
                "nothing downstream will notice.",
                CLAUDE_LINEAGE,
            )

    @property
    def name(self) -> str:
        """`claude-code/<the model that answered>`, resolved by asking if need be.

        The pipeline reads this once, before its loop, to find which subjects are already
        claimed -- so it is read before any draft exists and the honest answer is not yet
        known. Rather than label the run with the model that was *requested* (which the
        harness is free to ignore) or with a placeholder (which would reach the database),
        the first read spends one minimal call to find out.

        Cached afterwards, and re-checked on every call: see `_attribute`.
        """
        if self._answered_model is None:
            self._answered_model = self._probe_model()
        return f"{NAME_PREFIX}/{self._answered_model}"

    @property
    def answered_model(self) -> str | None:
        """The resolved model id, or None if nothing has been asked yet. No call made."""
        return self._answered_model

    def _probe_model(self) -> str:
        """One deliberately tiny call whose only product is a name.

        Small on purpose, and this is the opposite of what it looks like. The obvious
        design is to probe with the generator's real prompt, on the theory that the name
        should be resolved under the conditions the run will use. That was tried and it is
        worse: **the amount of harness side-work in an envelope scales with the request.**
        Measured, a two-word probe comes back with exactly one entry, and the same probe
        carrying the real ~850-token system prompt comes back with a `claude-haiku-4-5`
        entry beside the answering model -- which is precisely the ambiguity the name
        resolution must not have.

        The soundness argument does not need the big prompt anyway. `_run` has already
        refused any envelope without a textual `result`, so the session model demonstrably
        ran, so it demonstrably has a `modelUsage` entry. A single-entry envelope carrying
        a real answer therefore NAMES the session model -- it is not a guess between
        candidates, there is only one candidate and it is known to have answered.

        It goes through the same `build_argv` as a real call, so the sandbox and the
        configuration are identical; only the prompt is smaller.
        """
        return answering_model(
            self._call_raw(system="Answer with the single word: ok.", user="ok"),
            requested=self._requested_model,
        )

    def _call_raw(self, *, system: str, user: str) -> dict[str, object]:
        return _run(
            build_argv(cli=self._cli, system=system, user=user, model=self._requested_model),
            timeout=self._timeout,
            role=self._role,
            outage_detail=self._outage_detail,
        )

    def _call(self, *, system: str, user: str) -> str:
        """One call, returning the model's raw text, having pinned who wrote it."""
        envelope = self._call_raw(system=system, user=user)
        self._attribute(envelope)
        return str(envelope.get("result") or "")

    def _attribute(self, envelope: Mapping[str, object]) -> None:
        """Pin the answering model, and refuse to continue if it stopped appearing.

        The first answer fixes the name. A later response in which that model does not
        appear at all means the harness fell back, and every row already written -- plus
        every row still to come, since `pipeline.learn` captured `name` once before its
        loop -- would be filed under a model that did not write it. There is no repair for
        that after the fact and nothing downstream can see it, so the run stops there.

        `answering_model` owns the rule; this owns the state. The split matters because
        the rule has to be testable against one envelope, and the bug it replaced was only
        visible on a run of a hundred and fifty.
        """
        if self._answered_model is None:
            self._answered_model = answering_model(envelope, requested=self._requested_model)
        else:
            answering_model(envelope, expected=self._answered_model)
        self._warn_about_new_companions(envelope)

    def _warn_about_new_companions(self, envelope: Mapping[str, object]) -> None:
        """One warning per model id that turns up beside this run's own, and is new.

        Extra entries are harness overhead, and treating them as such is what makes the
        membership rule sound. It is still an assumption. A model that has never been seen
        on this run appearing now is either new overhead or a co-author, and this module
        cannot tell those apart from one envelope -- so it says so once, rather than
        silently, and does not stop a run over it.
        """
        for model_id in _entry_identities(envelope):
            if model_id == self._answered_model or model_id in self._auxiliary_models:
                continue
            self._auxiliary_models.add(model_id)
            logger.warning(
                "%r answered alongside %r on this run. It is being treated as harness "
                "overhead, which is what the attribution rule assumes -- but this module "
                "cannot tell overhead from a co-author out of one response.",
                model_id,
                self._answered_model,
            )


class ClaudeCodeClaimGenerator(_ClaudeCodeBackend):
    """`ClaimGenerator` backed by the `claude` CLI. The sibling of `OllamaClaimGenerator`.

    Sends the prompt `build_generation_prompt` builds, unchanged, and parses the reply
    with `parse_draft`, unchanged. Everything that makes this class interesting is in the
    module docstring: how the (system, user) split survives a CLI that takes one prompt,
    why the tool boundary is `--tools ""` rather than a deny list, why `name` is read out
    of the response, and which envelope shape lies about having succeeded.

    **The model choice does not collide with the faithfulness judge today, and that is
    checked rather than remembered.** The judge is `qwen3.5:9b`; this is a Claude model;
    `collides_with_judge` is consulted at construction against `CLAUDE_LINEAGE` so that
    the day the judge moves to a Claude model, constructing this warns that faithfulness
    has stopped being an audit. The lineage word is used instead of a model id because the
    hazard belongs to the vendor -- `claude-opus-5` grading `claude-haiku-4-5` shares a
    post-training pipeline and a set of blind spots -- and because the real model id is
    not known until something has been asked.

    **There is no `release()`.** Its ollama siblings hold VRAM the judge needs back; this
    holds nothing local. An empty `release()` would invite a caller to believe otherwise.
    """

    _role = "claude claim generator"
    _outage_detail = (
        "No claim was drafted and none was refused -- returning an empty draft here "
        "would record 'these spans establish nothing' against every symbol in the repo "
        "because the CLI could not reach a model."
    )

    def __init__(
        self,
        model: str | None = None,
        *,
        cli: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        kind: str = DEFAULT_KIND,
    ) -> None:
        super().__init__(model, cli=cli, timeout=timeout)
        self._kind = kind

    def draft(self, *, subject: str, offered: Sequence[Offer]) -> Draft:
        """Draft one claim about `subject` from the offered spans, citing by number.

        The same three outcomes as `OllamaClaimGenerator.draft`, for the same reasons: an
        empty `Draft` when the model abstained, when it answered unreadably, and when
        there was nothing to offer it; `GeneratorUnavailable` when the CLI did not answer
        at all. References are whatever integers came back, not a validated subset of the
        menu -- `Draft.resolve` is what drops off-menu references and reports them, and
        repairing them here would delete the measurement this seam exists to produce.
        """
        if not offered:
            # Not sent. Every claim from an empty menu would be uncited, so the only
            # answer that is not a fabrication is the empty one -- and asking anyway
            # spends a real call to invite a guess.
            logger.debug("no offers for %s; refusing without consulting the model", subject)
            return Draft(claim="", cited_refs=(), kind=self._kind)

        system, user = build_generation_prompt(subject=subject, offered=offered)
        return parse_draft(self._call(system=system, user=user), kind=self._kind)


class ClaudeCodePurposeModel(_ClaudeCodeBackend):
    """`generate.purpose.PurposeModel` behind the CLI: strings in, string out.

    The sibling of `OllamaPurposeModel`, and the narrowness of the seam is the safety
    property rather than a simplification. `purpose.py` runs a `docstring_blind` condition
    that strips the docstring before the prompt is built and never hands a backend the
    `SourceView`, so nothing here holds a pointer back to the file whose docstring was
    just removed.

    That is a stronger guarantee for a local backend than for this one, and the difference
    is the reason this module exists in the shape it does: an ollama backend cannot open a
    file because it is a POST to a model, while a subprocess running an agent CLI could
    open the file, re-read the stripped docstring, and read `git log` to find the held-out
    gold label itself -- scoring the blind condition on the answer. `--tools ""` is what
    closes that, and the live test in `tests/test_claude_code.py` is what checks it
    actually closed rather than merely was requested.

    Returns the model's raw text. `normalise_purpose` in `purpose.py` is what reduces it
    to something `token_f1` can compare, and it is applied to the shuffled control by the
    same code path -- which is the reason a chatty answer is tolerated here.
    """

    _role = "claude purpose model"
    _outage_detail = (
        "No purpose was inferred. Returning an empty string here would be scored by "
        "token-F1 as a legitimate wrong answer, and a run made while the CLI was broken "
        "would read as a model that understands nothing."
    )

    def complete(self, *, system: str, user: str) -> str:
        """One completion. Raises `GeneratorUnavailable` rather than returning `''`.

        The empty string is the one value this must never invent, for the reason its
        ollama sibling gives: `token_f1` scores it as a real answer with no overlap, so an
        outage would come out of the far end as a confident zero -- a number shaped
        exactly like a finding about the model that is in fact a fact about the harness.
        """
        return self._call(system=system, user=user)
