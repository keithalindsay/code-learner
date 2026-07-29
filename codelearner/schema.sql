-- code-learner index schema.
--
-- One DB file per repository (`.codelearner/index.db`). Cross-repo contamination is
-- prevented structurally: there is no shared store, and `meta.repo_root` pins this
-- file to exactly one repo root (see db.bind_repo_root).
--
-- The tier model is expressed in the schema itself rather than bolted on:
--   tier 0 FACT      -- parsed straight out of the source; reproducible from source alone
--   tier 1 RESOLVED  -- a name resolved to a symbol; may be ambiguous, carries confidence
--   tier 2 INFERRED  -- LLM-asserted; lives in `assertions` and must be evidence-bound
--
-- An edge is the clearest illustration: seeing the call site `foo()` is a tier-0 fact,
-- but deciding *which* `foo` it refers to is a tier-1 resolution that can be wrong.
-- Both live in one row -- `dst_name` is always populated, `dst_symbol_id` only once
-- something resolved it -- so an unresolved edge is honestly represented rather than
-- dropped or guessed.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    -- Repo-root-relative, POSIX separators. Unique per repo.
    path         TEXT NOT NULL UNIQUE,
    lang         TEXT NOT NULL,
    -- sha256 of the whole file's bytes. Cheap re-index check.
    content_hash TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    -- st_mtime_ns at index time; the fast path for staleness detection.
    mtime_ns     INTEGER NOT NULL,
    indexed_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS symbols (
    id           INTEGER PRIMARY KEY,
    file_id      INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    -- 'module' | 'class' | 'function' | 'method'
    kind         TEXT NOT NULL,
    -- Bare name, e.g. `acquire`.
    name         TEXT NOT NULL,
    -- Dotted path from the module root, e.g. `swarmsync.blackboard.leases.acquire`.
    qualname     TEXT NOT NULL,
    -- Enclosing symbol (class for a method, module for a top-level function).
    parent_id    INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    -- 1-based inclusive line span, for human-facing citation.
    line_start   INTEGER NOT NULL,
    line_end     INTEGER NOT NULL,
    -- Byte span, for exact slicing without re-deriving line offsets.
    byte_start   INTEGER NOT NULL,
    byte_end     INTEGER NOT NULL,
    -- sha256 of exactly this symbol's source bytes. THE binding primitive: an
    -- assertion about this symbol expires when this hash changes.
    content_hash TEXT NOT NULL,
    docstring    TEXT,
    signature    TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_symbols_qualname ON symbols(qualname);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_id);

CREATE TABLE IF NOT EXISTS edges (
    id            INTEGER PRIMARY KEY,
    src_symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    -- 'calls' | 'imports' | 'inherits'
    kind          TEXT NOT NULL,
    -- The name as written at the call/import/base-class site. ALWAYS populated --
    -- this is the tier-0 fact and it never depends on resolution succeeding.
    dst_name      TEXT NOT NULL,
    -- For imports: the name this binds in the importing module (`import x as y`
    -- binds `y`). NULL for other edge kinds. Without it, alias-qualified calls
    -- like `events_mod.tail()` cannot be matched to their target.
    local_name    TEXT,
    -- NULL until a resolver binds it. Tier-1.
    dst_symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    -- 0 while unresolved (the call site alone), 1 once resolved to a symbol.
    tier          INTEGER NOT NULL DEFAULT 0,
    -- Resolver confidence in [0,1]; NULL while unresolved.
    confidence    REAL,
    -- Which resolver bound it, so a bad resolver's edges can be found and requeued.
    resolver      TEXT,
    -- 1-based line of the reference site itself.
    line          INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_symbol_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_symbol_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst_name ON edges(dst_name);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);
