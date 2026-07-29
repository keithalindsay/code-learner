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
    -- Derived deterministically from path conventions, so it is a tier-0 fact.
    -- Exists because a test and the code it tests are different KINDS of answer:
    -- measured on swarm-sync, both retrieval modalities rank tests above the
    -- implementations they exercise. Recording which is which makes that
    -- measurable, and lets a caller say which kind it wants.
    is_test      INTEGER NOT NULL DEFAULT 0,
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

-- Retrieval units. One chunk per symbol, cut on symbol boundaries rather than a
-- fixed window: splitting a function in half produces two fragments that are each
-- individually misleading, and the embedding of half a function is not the
-- embedding of anything a question would ever be about.
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    symbol_id    INTEGER NOT NULL UNIQUE REFERENCES symbols(id) ON DELETE CASCADE,
    -- Retrieval text: a generated context header followed by the source slice. The
    -- header exists because a bare method body is not self-describing -- `def
    -- acquire(...)` retrieved alone gives a reader no idea which class or module
    -- it belongs to, and the embedding loses that signal too.
    text         TEXT NOT NULL,
    -- Header alone, so a caller can show provenance without re-deriving it.
    header       TEXT NOT NULL,
    char_count   INTEGER NOT NULL,
    -- Hash of `text`. Lets a re-index skip re-embedding chunks that did not change,
    -- which is the difference between a 3-second and a 3-minute update.
    text_hash    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_id);

-- Lexical retrieval modality. `content=''` makes this an external-content index:
-- FTS5 stores only the inverted index, not a second copy of every chunk's text.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 0'
);

-- Keep the FTS index in step with the table it mirrors. Without these an edited or
-- deleted chunk leaves a phantom entry that keeps matching queries forever.
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Tier 2: INFERRED claims, and the machinery that makes one accountable.
--
-- An LLM's statement about code has a failure mode the parsed tiers do not have:
-- it is unverifiable when it is read, and it goes stale without saying so. A
-- rationale that was true in March is still served, word for word and just as
-- confidently, in June. That single property is why the honest competitors refuse
-- inference outright ("structural facts only") -- not because the claims are
-- useless, but because nothing about them is checkable.
--
-- Four tables, each removing one part of that:
--
--   assertions      -- the claim and its status. Rows are never deleted.
--   evidence_spans  -- the exact bytes it cites. No spans, no assertion.
--   verdicts        -- what a judge concluded, INCLUDING every refusal.
--   staleness_log   -- which citation stopped matching disk, and when.
--
-- The subject is recorded the way `edges` records its target, and for the same
-- reason: `subject_qualname` is always populated and durable, `subject_symbol_id`
-- is the resolved convenience link and may be NULL. That is not symmetry for its
-- own sake. An `ON DELETE CASCADE` to symbols(id) would mean a routine re-index
-- silently deletes the entire assertion store, because re-indexing replaces symbol
-- rows wholesale. The name survives that; the row id does not.
CREATE TABLE IF NOT EXISTS assertions (
    id                INTEGER PRIMARY KEY,
    -- Dotted path of the symbol the claim is ABOUT, e.g. `codelearner.db.init_db`.
    subject_qualname  TEXT NOT NULL,
    -- Resolved link, when the subject is in the graph right now. SET NULL rather
    -- than CASCADE, per the note above: losing the link must not lose the claim.
    subject_symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    -- 'purpose' | 'invariant' | 'risk' | ... -- what KIND of claim this is. Worth a
    -- column because "what is this for" and "what must stay true here" have
    -- different readers and very different costs when wrong, and a caller should be
    -- able to take one and refuse the other.
    kind              TEXT NOT NULL,
    claim             TEXT NOT NULL,
    -- 'active'   -- admitted; servable for as long as its evidence still hashes.
    -- 'rejected' -- a judge refused it. RETAINED, never deleted: the rejections are
    --               the only evidence that the gate does anything at all, and a
    --               store that deletes them can report whatever pass rate it likes.
    -- 'stale'    -- a cited span changed underneath it. Also retained, because it
    --               names exactly what is worth re-deriving and what it used to say.
    -- The CHECK belongs here rather than in Python because a typo'd status is
    -- otherwise indistinguishable from a rejected one -- both merely fail to be
    -- 'active', so the claim stops being served with no record of why.
    status            TEXT NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active', 'rejected', 'stale')),
    -- Model id + prompt version that produced this. The unit of recall: when a
    -- generator turns out to hallucinate one particular shape of claim, this makes
    -- "find everything it wrote" a query rather than a re-run of the whole pipeline.
    generator         TEXT,
    -- The generator's own confidence in [0,1]. Advisory, and deliberately NOT an
    -- input to servability: a model's confidence in an uncited claim is a number
    -- about the model, not about the code.
    confidence        REAL,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- When `status` last moved. Answers "how long were we serving that" after the
    -- fact, which is the first question asked once a bad claim is found.
    status_changed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_assertions_subject ON assertions(subject_qualname);
CREATE INDEX IF NOT EXISTS idx_assertions_symbol ON assertions(subject_symbol_id);
CREATE INDEX IF NOT EXISTS idx_assertions_status ON assertions(status);

-- The citations. This table is the whole reason a tier-2 claim is checkable: it
-- records the exact bytes the claim was derived from, so a reader can go look, and
-- so the store can re-hash them at serve time and notice when they have moved.
--
-- `path` is a plain repo-relative string and NOT a reference to files(id). That is
-- deliberate and it is the subtlest rule in this schema. A file dropped from the
-- index -- renamed, deleted, newly gitignored -- would cascade its spans away, and
-- an assertion that loses its last span does not become unsupported. It becomes
-- VACUOUSLY supported, because "every cited span still matches" is trivially true
-- of no spans at all. Storing the path as text turns that case into a span whose
-- file is missing, which is a staleness event with a reason attached instead of a
-- silent promotion. (The reader checks for an empty evidence set anyway; a rule
-- this easy to get wrong deserves both.)
CREATE TABLE IF NOT EXISTS evidence_spans (
    id           INTEGER PRIMARY KEY,
    assertion_id INTEGER NOT NULL REFERENCES assertions(id) ON DELETE CASCADE,
    -- Repo-root-relative, POSIX separators -- the same coordinate space as
    -- `files.path`, so a citation reads the same whether or not the file is indexed.
    path         TEXT NOT NULL,
    -- 1-based inclusive lines, for the human-facing `file:line` citation. Derived
    -- FROM the byte range at write time rather than accepted alongside it: a
    -- citation whose line numbers disagree with its bytes is one nobody can check,
    -- and that disagreement would never surface on its own.
    line_start   INTEGER NOT NULL,
    line_end     INTEGER NOT NULL,
    -- Byte span, for re-slicing exactly the evidence without re-deriving offsets.
    byte_start   INTEGER NOT NULL,
    byte_end     INTEGER NOT NULL,
    -- sha256 of exactly source[byte_start:byte_end] as it read at write time (see
    -- ingest.types.content_hash). THE expiry primitive. Explicitly not the whole
    -- file's hash: an unrelated edit elsewhere in a 2,000-line module must not
    -- expire a claim about one function in it, or staleness becomes noise and the
    -- first thing anyone does with noise is stop reading it.
    content_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_assertion ON evidence_spans(assertion_id);
CREATE INDEX IF NOT EXISTS idx_evidence_path ON evidence_spans(path);

-- Adjudication. A judge is asked to REFUTE a claim using only the spans that claim
-- cites -- a different question from "does this look right", and the only one with
-- a checkable answer, because the evidence it may use is finite and written down.
--
-- Every verdict is kept, the refusals most of all. A pipeline that deletes what it
-- rejected cannot show its gate does anything, and cannot tell a generator that got
-- better from a judge that got lazier -- the two look identical from the pass rate.
CREATE TABLE IF NOT EXISTS verdicts (
    id           INTEGER PRIMARY KEY,
    assertion_id INTEGER NOT NULL REFERENCES assertions(id) ON DELETE CASCADE,
    -- Judge identity (model + prompt version), for the same recall reason as
    -- `assertions.generator`.
    judge        TEXT NOT NULL,
    -- 'supported'   -- the cited spans do establish the claim.
    -- 'refuted'     -- the cited spans contradict it.
    -- 'unsupported' -- the spans neither establish nor contradict it. Kept distinct
    --                  from 'refuted' on purpose: "the evidence is silent" is a
    --                  fixable citation problem and "the evidence says otherwise" is
    --                  a wrong claim, and collapsing them hides which one a
    --                  generator actually has. Both stop it being served.
    verdict      TEXT NOT NULL
                 CHECK (verdict IN ('supported', 'refuted', 'unsupported')),
    -- The judge's reasoning, in its own words. Kept because the rejection log is
    -- only useful if someone can read WHY, and because a judge that rejects for
    -- consistently bad reasons is a thing that has to be discoverable.
    rationale    TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_verdicts_assertion ON verdicts(assertion_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_verdict ON verdicts(verdict);

-- The stat baseline for the two-stage freshness check (see assertions/stale.py).
--
-- Re-hashing every cited span on every query is O(spans) file reads per query. The
-- fast path is a stat(): if a span's file has the same mtime_ns AND the same size as
-- when the span was last hash-verified, the bytes are taken to be unchanged and no
-- read happens. Only a moved mtime (or a changed size) falls through to the full
-- re-hash. Measured on this repo's own index, the steady state is one stat() per
-- distinct cited FILE per query and zero reads.
--
-- This table is separate from `evidence_spans` rather than three more columns on it,
-- because the two have opposite lifetimes. A citation is the immutable thing the
-- claim was admitted on -- path, byte range, hash -- and nothing at serve time should
-- be writing to that row. This is the mutable observation ABOUT that citation, and it
-- is disposable: delete every row here and the only consequence is that the next
-- query re-hashes, which is exactly the pre-fast-path behaviour. The CASCADE matters
-- for the same reason -- re-citing an assertion replaces its span rows, and a stat
-- baseline that outlived the hash it was a shortcut for would authorise skipping the
-- read for a span nobody ever verified.
--
-- `files.mtime_ns` is deliberately NOT reused for this. That column records when a
-- file was last INDEXED, which is a different event from when a citation was last
-- verified, and it exists only for files currently in the index -- a span may cite a
-- file that was never indexed or has since been dropped. Sharing one column between
-- the two would make a re-index look like a verification.
CREATE TABLE IF NOT EXISTS span_verifications (
    -- One row per span, so a span that has NEVER been hash-verified simply has no
    -- row and cannot take the fast path. Per-file would be wrong here: it would let
    -- a brand-new citation inherit a neighbour's baseline and be served without its
    -- own hash ever having been checked once.
    span_id       INTEGER PRIMARY KEY REFERENCES evidence_spans(id) ON DELETE CASCADE,
    -- st_mtime_ns and st_size of the cited file at that verification. Both, not just
    -- mtime: size is free once the file is stat'd, and a same-second edit that
    -- changes length is caught by it.
    mtime_ns      INTEGER NOT NULL,
    size_bytes    INTEGER NOT NULL,
    -- The hash actually observed on disk at that verification. Stored (rather than
    -- implied by evidence_spans.content_hash) so the row states what it witnessed:
    -- the fast path requires this to still equal the cited hash, so a baseline can
    -- never vouch for a hash it did not see.
    verified_hash TEXT NOT NULL,
    -- When those bytes were last actually read and hashed. THE honest freshness
    -- number: a claim confirmed by stat() alone is only as fresh as this timestamp,
    -- and every served assertion carries it so a caller can see that for itself.
    verified_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Expiry events. Staleness is detected on READ -- the check is a re-hash of the
-- cited bytes at the moment something asks for the claim, not a background sweep --
-- so without this table the event would exist only as a status flip, with nothing
-- recording which citation moved or when it was noticed.
--
-- One row per transition, not per read: a stale assertion is never re-checked
-- (`status` alone excludes it from the serving path), so the log grows only when
-- something actually expires. That makes its growth rate a real signal about how
-- fast this repo invalidates what was inferred about it.
CREATE TABLE IF NOT EXISTS staleness_log (
    id            INTEGER PRIMARY KEY,
    assertion_id  INTEGER NOT NULL REFERENCES assertions(id) ON DELETE CASCADE,
    -- The span that failed. A plain integer, not a reference: re-citing an assertion
    -- replaces its span rows, and a log has to outlive the rows it describes or it
    -- stops being a record of what happened.
    span_id       INTEGER,
    -- 'hash_mismatch'  -- the file is there; the cited bytes are not what they were.
    -- 'file_missing'   -- the cited file is gone from disk entirely.
    -- 'span_truncated' -- the file is now shorter than the cited byte range.
    -- 'no_evidence'    -- the assertion has no spans at all. Unreachable if the
    --                     write gate holds, and checked anyway, because the failure
    --                     it guards against is serving a claim on the strength of an
    --                     empty evidence set -- which reads as success everywhere.
    reason        TEXT NOT NULL,
    -- What was cited vs. what is on disk now. Both nullable: a missing file has no
    -- observed hash, and that absence is itself the finding.
    expected_hash TEXT,
    observed_hash TEXT,
    detected_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_staleness_assertion ON staleness_log(assertion_id);
