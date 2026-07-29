"""tree-sitter extraction of symbols and reference edges from Python source.

Scope is deliberately narrow: this produces tier-0 facts only -- what the parser
can see in one file, with no cross-file reasoning. `helper(x)` yields an edge whose
`dst_name` is the literal text `helper`; deciding *which* `helper` that is belongs
to a resolver and is tier-1. Keeping that line sharp here is what makes the tier
model honest downstream instead of decorative.
"""
from __future__ import annotations

from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from .types import (
    EDGE_CALLS,
    EDGE_IMPORTS,
    EDGE_INHERITS,
    KIND_CLASS,
    KIND_FUNCTION,
    KIND_METHOD,
    KIND_MODULE,
    Edge,
    FileExtract,
    Symbol,
    content_hash,
)

_LANGUAGE = Language(tspython.language())


def _parser() -> Parser:
    # Parsers are cheap and not documented as thread-safe; make one per call rather
    # than sharing a module-global that would become a latent concurrency bug the
    # first time indexing is parallelised.
    return Parser(_LANGUAGE)


def module_qualname(rel_path: str) -> str:
    """Map a repo-relative path to a dotted module name.

    `pkg/sub/mod.py` -> `pkg.sub.mod`; `pkg/sub/__init__.py` -> `pkg.sub`.
    """
    parts = list(_path_parts(rel_path))
    if not parts:
        return ""
    last = parts[-1]
    if last.endswith(".py"):
        stem = last[:-3]
        if stem == "__init__":
            parts = parts[:-1]
        else:
            parts[-1] = stem
    return ".".join(parts)


def _path_parts(rel_path: str) -> tuple[str, ...]:
    """Split a POSIX-style relative path into parts, tolerating either separator."""
    return tuple(p for p in rel_path.replace("\\", "/").split("/") if p and p != ".")


def _text(src: bytes, node: Node) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _docstring(src: bytes, body: Node | None) -> str | None:
    """Return the docstring of a body block, if its first statement is a string."""
    if body is None:
        return None
    for child in body.named_children:
        if child.type == "expression_statement" and child.named_children:
            inner = child.named_children[0]
            if inner.type == "string":
                raw = _text(src, inner)
                return raw.strip("\"'").strip() or None
        # Only the FIRST statement can be a docstring.
        return None
    return None


def _signature(src: bytes, node: Node) -> str | None:
    """Reconstruct `name(params) -> ret` without dragging in the body."""
    name = node.child_by_field_name("name")
    params = node.child_by_field_name("parameters")
    if name is None:
        return None
    sig = _text(src, name)
    if params is not None:
        sig += _text(src, params)
    ret = node.child_by_field_name("return_type")
    if ret is not None:
        sig += " -> " + _text(src, ret)
    return sig


def extract(source: bytes, rel_path: str, mtime_ns: int = 0) -> FileExtract:
    """Extract every symbol and reference edge from one Python file.

    Never raises on malformed input: tree-sitter is error-tolerant by design, and a
    repo with one unparseable file should still index the other nine hundred.
    """
    tree = _parser().parse(source)
    root = tree.root_node

    mod_qual = module_qualname(rel_path)
    result = FileExtract(
        path=rel_path,
        lang="python",
        content_hash=content_hash(source),
        size_bytes=len(source),
        mtime_ns=mtime_ns,
    )
    result.symbols.append(
        Symbol(
            kind=KIND_MODULE,
            name=mod_qual.rsplit(".", 1)[-1] if mod_qual else rel_path,
            qualname=mod_qual,
            line_start=1,
            line_end=max(1, root.end_point[0] + 1),
            byte_start=0,
            byte_end=len(source),
            content_hash=content_hash(source),
            parent_qualname=None,
            docstring=_docstring(source, root),
        )
    )

    # (qualname, kind) of the enclosing definition. Module sits at the bottom so
    # module-level calls and imports have a real source symbol rather than being
    # silently dropped.
    scope: list[tuple[str, str]] = [(mod_qual, KIND_MODULE)]

    def visit(node: Node) -> None:
        node_type = node.type

        if node_type in ("function_definition", "class_definition"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                for child in node.children:
                    visit(child)
                return
            name = _text(source, name_node)
            parent_qual, parent_kind = scope[-1]
            qual = f"{parent_qual}.{name}" if parent_qual else name

            if node_type == "class_definition":
                kind = KIND_CLASS
            else:
                kind = KIND_METHOD if parent_kind == KIND_CLASS else KIND_FUNCTION

            body = node.child_by_field_name("body")
            result.symbols.append(
                Symbol(
                    kind=kind,
                    name=name,
                    qualname=qual,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    byte_start=node.start_byte,
                    byte_end=node.end_byte,
                    content_hash=content_hash(source[node.start_byte : node.end_byte]),
                    parent_qualname=parent_qual or None,
                    docstring=_docstring(source, body),
                    signature=_signature(source, node) if kind != KIND_CLASS else None,
                )
            )

            if node_type == "class_definition":
                supers = node.child_by_field_name("superclasses")
                if supers is not None:
                    for base_node in supers.named_children:
                        # Skip keyword args like `metaclass=ABCMeta`.
                        if base_node.type == "keyword_argument":
                            continue
                        result.edges.append(
                            Edge(
                                src_qualname=qual,
                                kind=EDGE_INHERITS,
                                dst_name=_text(source, base_node),
                                line=base_node.start_point[0] + 1,
                            )
                        )

            scope.append((qual, kind))
            for child in node.children:
                visit(child)
            scope.pop()
            return

        if node_type == "call":
            func = node.child_by_field_name("function")
            if func is not None:
                result.edges.append(
                    Edge(
                        src_qualname=scope[-1][0],
                        kind=EDGE_CALLS,
                        dst_name=_text(source, func),
                        line=node.start_point[0] + 1,
                    )
                )
            # Fall through: arguments may contain nested calls and lambdas.

        elif node_type == "import_statement":
            # `import a.b.c` binds `a`; `import a.b.c as x` binds `x`.
            for child in node.named_children:
                target, alias = _import_target(source, child)
                result.edges.append(
                    Edge(
                        src_qualname=scope[-1][0],
                        kind=EDGE_IMPORTS,
                        dst_name=target,
                        line=node.start_point[0] + 1,
                        local_name=alias or target.split(".")[0],
                    )
                )

        elif node_type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            module_path = _text(source, module_node) if module_node is not None else ""
            names = [
                c
                for c in node.named_children
                if c is not module_node
                and c.type in ("dotted_name", "aliased_import", "wildcard_import")
            ]
            if not names:
                result.edges.append(
                    Edge(
                        scope[-1][0],
                        EDGE_IMPORTS,
                        module_path,
                        node.start_point[0] + 1,
                        module_path.split(".")[0],
                    )
                )
            for child in names:
                target, alias = _import_target(source, child)
                if target == "*":
                    continue  # binds unknown names; nothing honest to record
                result.edges.append(
                    Edge(
                        src_qualname=scope[-1][0],
                        kind=EDGE_IMPORTS,
                        dst_name=f"{module_path}.{target}" if module_path else target,
                        line=node.start_point[0] + 1,
                        local_name=alias or target.split(".")[-1],
                    )
                )
            return

        for child in node.children:
            visit(child)

    for child in root.children:
        visit(child)

    return result


def _import_target(source: bytes, node: Node) -> tuple[str, str | None]:
    """Return `(target_path, alias_or_None)` for one imported name.

    Both halves are needed: the target says what was imported, the alias says what
    name the surrounding code will actually call it by.
    """
    if node.type == "aliased_import":
        name_node = node.child_by_field_name("name")
        alias_node = node.child_by_field_name("alias")
        target = _text(source, name_node) if name_node is not None else _text(source, node)
        alias = _text(source, alias_node) if alias_node is not None else None
        return target, alias
    if node.type == "wildcard_import":
        return "*", None
    return _text(source, node), None


def extract_file(path: Path, repo_root: Path) -> FileExtract:
    """Read and extract one file on disk, recording its mtime for staleness checks."""
    stat = path.stat()
    rel = path.relative_to(repo_root).as_posix()
    return extract(path.read_bytes(), rel, mtime_ns=stat.st_mtime_ns)
