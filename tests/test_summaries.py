"""T31-T46: Summaries CRUD, profile scoping, schema."""

import logging
import os
import shutil
import sqlite3

from conftest import _make_summaries, _cleanup_db


def test_t31():
    """Summaries — add + idempotency."""
    sb = _make_summaries("t31")
    try:
        r1 = sb.add("https://example.com/page", "web", "Test Page",
                    highlights=["key point"], full_text="full summary text")
        r2 = sb.add("https://example.com/page", "web", "Test Page",
                    highlights=["key point"], full_text="full summary text")
        assert r2["already_exists"] is True
        assert r2["uuid"] == r1["uuid"]
    finally:
        sb.close(); _cleanup_db("t31"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t32():
    """Summaries — update + delete."""
    sb = _make_summaries("t32")
    try:
        r = sb.add("https://example.com/page2", "web", "Page 2",
                   highlights=["old"], full_text="text")
        assert sb.update(r["uuid"], profile="all", highlights=["new"])
        got = sb.get(r["uuid"], profile="all")
        assert "new" in got["highlights"]
        assert sb.delete(r["uuid"], profile_name="any", profile="all")
        assert sb.get(r["uuid"], profile="all") is None
        assert not os.path.exists(r["summary_path"])
    finally:
        sb.close(); _cleanup_db("t32"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t33():
    """Summaries — search + list."""
    sb = _make_summaries("t33")
    try:
        sb.add("https://example.com/a", "web", "Alpha", highlights=["alpha"], full_text="a text")
        sb.add("https://example.com/b", "youtube", "Beta", highlights=["beta"], full_text="b text")
        assert len(sb.search("alpha", profile="all")) >= 1
        assert sb.list_summaries(source_type="web", limit=5, profile="all")["total"] >= 1
    finally:
        sb.close(); _cleanup_db("t33"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t34():
    """Summaries — expire + sync."""
    sb = _make_summaries("t34")
    try:
        sb.add("https://example.com/old", "web", "Old", highlights=["old"], full_text="old text")
        assert len(sb.list_expiring(max_age_days=0, profile="all")) >= 1
        assert isinstance(sb.sync(), dict)
    finally:
        sb.close(); _cleanup_db("t34"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t35():
    """Summaries — get + batch delete."""
    sb = _make_summaries("t35")
    try:
        r1 = sb.add("https://example.com/g1", "web", "G1", highlights=["g1"], full_text="text1")
        r2 = sb.add("https://example.com/g2", "web", "G2", highlights=["g2"], full_text="text2")
        got = sb.get(r1["uuid"], profile="all")
        assert got is not None and "summary_path" in got
        result = sb.delete_multiple([r1["uuid"], r2["uuid"], "nonexistent"],
                                    profile_name="any", profile="all")
        assert result["deleted"] == 2 and result["not_found"] == 1
    finally:
        sb.close(); _cleanup_db("t35"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t36():
    """Obsidian ingest."""
    import tempfile
    from conftest import _make_backend, _cleanup_db, _cleanup_qdrant_coll
    be = _make_backend("t36")
    try:
        assert hasattr(be, "ingest_obsidian")
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "test.md"), "w") as f:
            f.write("---\ntitle: Test\n---\n\nContent.")
        result = be.ingest_obsidian(tmp)
        assert result.get("ingested", 0) >= 1
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t36")


def test_t37():
    """Dedup at add time."""
    from conftest import _make_backend, _cleanup_db, _cleanup_qdrant_coll
    be = _make_backend("t37")
    try:
        be.add("RTX 4090 GPU with 24GB VRAM", data_type="ENV-DATA", data_id="hw")
        result = be.add("RTX 4090 GPU 24GB VRAM", data_type="ENV-DATA", data_id="hw")
        assert isinstance(result, dict)
        assert result.get("status") in ("duplicate", "possible_duplicate", "contradiction")
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t37")


def test_t38():
    """Low-trust archive."""
    from conftest import _make_backend, _cleanup_db, _cleanup_qdrant_coll
    be = _make_backend("t38")
    try:
        uuid = be.add("low trust", data_type="ENV-DATA", data_id="hw", trust_score=0.5)
        for _ in range(5):
            be.feedback(uuid, helpful=False)
        assert be._get_record(uuid)["trust_score"] < 0.3
        result = be.sleep(max_items=50, min_age_hours=0)
        assert isinstance(result, dict)
        assert "archived" in result and "ttl_expired" in result
        assert "merged" not in result, "sleep() does not do content merging"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t38")


def test_t39():
    """Backup — atomic SQLite copy."""
    from conftest import _make_backend, _cleanup_db, _cleanup_qdrant_coll
    be = _make_backend("t39")
    try:
        be.add("backup test", data_type="CUSTOM")
        result = be.backup()
        assert "backup_path" in result and os.path.exists(result["backup_path"])
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t39")


def test_t40():
    """Backup — custom directory."""
    import tempfile
    from conftest import _make_backend, _cleanup_db, _cleanup_qdrant_coll
    be = _make_backend("t40")
    try:
        be.add("backup custom dir", data_type="CUSTOM")
        tmp = tempfile.mkdtemp(prefix="hlm-backup-t40-")
        result = be.backup(dest_dir=tmp)
        assert os.path.exists(result["backup_path"])
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t40")


def test_t41():
    """Backup — old backup cleanup (provider layer feature)."""
    import tempfile
    from conftest import _make_backend, _cleanup_db, _cleanup_qdrant_coll
    be = _make_backend("t41")
    try:
        be.add("backup cleanup", data_type="CUSTOM")
        tmp = tempfile.mkdtemp(prefix="hlm-backup-t41-")
        result = be.backup(dest_dir=tmp)
        assert os.path.exists(result["backup_path"])
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t41")


def test_t42():
    """Profile-scoped summaries — cross-profile protect."""
    sb = _make_summaries("t42")
    try:
        r = sb.add("https://example.com/prof", "web", "Profile", highlights=["h"],
                   full_text="text", profile_name="profile-a")
        assert sb.delete(r["uuid"], profile_name="profile-b", profile="own") is False
        assert sb.get(r["uuid"], profile="all") is not None
    finally:
        sb.close(); _cleanup_db("t42"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t43():
    """Profile-scoped summaries — delete with all override."""
    sb = _make_summaries("t43")
    try:
        r = sb.add("https://example.com/all", "web", "All", highlights=["h"],
                   full_text="text", profile_name="profile-a")
        assert sb.delete(r["uuid"], profile_name="profile-b", profile="all") is True
        assert sb.get(r["uuid"], profile="all") is None
    finally:
        sb.close(); _cleanup_db("t43"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t44():
    """Profile-scoped summaries — sync scoped to own."""
    sb = _make_summaries("t44")
    try:
        sb.add("https://example.com/own", "web", "Own", highlights=["h"],
               full_text="text", profile_name="own")
        assert isinstance(sb.sync(profile_name="own", profile="own"), dict)
    finally:
        sb.close(); _cleanup_db("t44"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t45():
    """Profile-scoped summaries — list with profile filter."""
    sb = _make_summaries("t45")
    try:
        sb.add("https://example.com/a", "web", "A", highlights=["a"],
               full_text="text", profile_name="p1")
        sb.add("https://example.com/b", "web", "B", highlights=["b"],
               full_text="text", profile_name="p2")
        # `profile` is the own/all scope word since 0.7.52; a bare profile
        # *name* here is the old calling convention and is now refused. Same
        # intent, expressed the way every other entry point expresses it.
        assert sb.list_summaries(profile_name="p1", profile="own")["total"] == 1
    finally:
        sb.close(); _cleanup_db("t45"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t46():
    """Schema version verification."""
    sb = _make_summaries("t46")
    try:
        row = sb._conn.execute("SELECT value FROM schema_info WHERE key='version'").fetchone()
        assert row and int(row[0]) >= 5
    finally:
        sb.close(); _cleanup_db("t46"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)

def test_t340():
    """The v6 UNIQUE migration resolves pre-existing duplicates instead of looping.

    `CREATE UNIQUE INDEX` cannot run while the table already violates it, and
    the attempt was wrapped in `except Exception: pass` commented "Index already
    exists or conflict" — so a real conflict was treated as a no-op. The
    verification then correctly found the index missing, declined to advance the
    version, and logged an error; every later init repeated the cycle. Observed
    on a production database stuck at v5 since 2026-08-06, having logged the
    same error 30 times, with the documented global UNIQUE on source_url absent
    the whole time — so `add()`'s own lookup was the only thing preventing
    duplicate sources.

    Builds a v5 database with duplicates and asserts the migration converges.
    """
    import sqlite3
    import sys
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from summaries import SummariesBackend

    tmp = tempfile.mkdtemp(prefix="hlm-t340-")
    db = os.path.join(tmp, "digests.db")
    try:
        # A v5-shaped database. Dropping the index is not enough: databases
        # created by current code declare `source_url TEXT NOT NULL UNIQUE`
        # inline, so the constraint survives. The databases that actually need
        # this migration predate that column definition — production's table
        # reads `source_url TEXT NOT NULL` — so the table has to be rebuilt
        # without it to reproduce the condition at all.
        seed = SummariesBackend(db, os.path.join(tmp, "files"))
        seed.close()
        c = sqlite3.connect(db)
        create_sql = c.execute(
            "SELECT sql FROM sqlite_master WHERE name='summaries'").fetchone()[0]
        legacy_sql = create_sql.replace("source_url TEXT NOT NULL UNIQUE",
                                        "source_url TEXT NOT NULL")
        assert legacy_sql != create_sql, "inline UNIQUE not found — schema changed"
        # The FTS triggers reference summaries; drop and let init recreate them.
        for trig in ("summaries_ai", "summaries_au", "summaries_ad"):
            c.execute(f"DROP TRIGGER IF EXISTS {trig}")
        c.execute("DROP INDEX IF EXISTS idx_summaries_source_url")
        c.execute("ALTER TABLE summaries RENAME TO summaries_legacy_tmp")
        c.execute(legacy_sql)
        c.execute("INSERT INTO summaries SELECT * FROM summaries_legacy_tmp")
        c.execute("DROP TABLE summaries_legacy_tmp")
        c.execute("INSERT OR REPLACE INTO schema_info(key, value) VALUES ('version', '5')")
        for uuid_, created in (("aaaa1111", "2026-01-01T00:00:00+00:00"),
                               ("bbbb2222", "2026-06-01T00:00:00+00:00"),
                               ("cccc3333", "2026-03-01T00:00:00+00:00")):
            c.execute("INSERT INTO summaries (uuid, source_url, source_type, title, "
                      "created_at, updated_at, status) VALUES (?,?,?,?,?,?,?)",
                      (uuid_, "https://example.invalid/dupe", "web", f"copy {uuid_}",
                       created, created, "complete"))
        c.commit()
        assert c.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 3
        c.close()

        # Re-open: the migration runs.
        sb = SummariesBackend(db, os.path.join(tmp, "files"))
        conn = sb._conn
        version = conn.execute(
            "SELECT value FROM schema_info WHERE key='version'").fetchone()[0]
        assert str(version) == "6", f"version did not advance: {version}"

        indexes = [r[1] for r in conn.execute("PRAGMA index_list(summaries)")]
        assert "idx_summaries_source_url" in indexes, "the UNIQUE index was not created"

        rows = conn.execute(
            "SELECT uuid FROM summaries WHERE source_url='https://example.invalid/dupe'"
        ).fetchall()
        assert len(rows) == 1, f"duplicates survived: {rows}"
        assert rows[0][0] == "bbbb2222", \
            f"kept {rows[0][0]} — the newest (bbbb2222, 2026-06-01) should win"

        # The constraint has to actually bite now, or the migration only moved
        # a version number.
        try:
            conn.execute("INSERT INTO summaries (uuid, source_url, source_type, title, "
                         "created_at, updated_at, status) VALUES (?,?,?,?,?,?,?)",
                         ("dddd4444", "https://example.invalid/dupe", "web", "x",
                          "2026-07-01T00:00:00+00:00", "2026-07-01T00:00:00+00:00",
                          "complete"))
            raise AssertionError("UNIQUE on source_url is not enforced")
        except sqlite3.IntegrityError:
            pass
        sb.close()

        # Idempotent: a second open must not remove anything or re-run the fix.
        sb2 = SummariesBackend(db, os.path.join(tmp, "files"))
        assert sb2._conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 1
        assert str(sb2._conn.execute(
            "SELECT value FROM schema_info WHERE key='version'").fetchone()[0]) == "6"
        sb2.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t341():
    """A blank or schemeless source_url does not become a shared identity.

    The canonical form *is* a summary's identity — it carries the UNIQUE
    constraint and decides `already_exists`. Two degenerate cases produced
    identities that were actively wrong:

        ""               -> ":///"              every blank input, one key
        "example.com/x"  -> "://example.com/x"
        "/local/path.md" -> ":///local/path.md" (the docstring accepts a path)
        None             -> TypeError

    urlparse puts a schemeless string entirely in `path`, and the generic branch
    then formatted `scheme://netloc` out of two empty strings. Two rows in the
    production database carried ':///' and '', and they are what blocked the v6
    UNIQUE migration (T340).
    """
    import tempfile
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from summaries import SummariesBackend

    tmp = tempfile.mkdtemp(prefix="hlm-t341-")
    try:
        sb = SummariesBackend(os.path.join(tmp, "d.db"), os.path.join(tmp, "files"))

        # Blank input is refused rather than given a shared identity.
        for blank in ("", "   ", None):
            try:
                got = sb._canonicalize_url(blank)
                raise AssertionError(f"{blank!r} canonicalized to {got!r} instead of raising")
            except ValueError:
                pass

        # Schemeless input keeps its meaning instead of gaining an empty scheme.
        assert sb._canonicalize_url("example.com/page") == "https://example.com/page"
        assert sb._canonicalize_url("/local/path.md") == "/local/path.md"
        assert sb._canonicalize_url("not a url") == "not a url"
        assert not sb._canonicalize_url("/a/one.md").startswith(":"), "path gained a scheme separator"
        assert sb._canonicalize_url("/a/one.md") != sb._canonicalize_url("/a/two.md"), \
            "distinct local paths collapsed onto one identity"

        # The existing normalisations are untouched.
        assert sb._canonicalize_url("HTTPS://Example.com/X/") == "https://example.com/X"
        assert sb._canonicalize_url("https://youtu.be/abc123") == \
            "https://www.youtube.com/watch?v=abc123"
        assert sb._canonicalize_url("https://github.com/o/r.git") == "https://github.com/o/r"
        assert sb._canonicalize_url("https://example.com/a?utm_source=x&id=7") == \
            "https://example.com/a?id=7"
        sb.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t343():
    """A misaligned FTS index is detected and rebuilt, not counted and ignored.

    `summaries_fts` is an external-content table: its rows read *through* to
    `summaries`, so `COUNT(*)` reports the content table's size whatever the
    inverted index actually holds. The init only compared those two counts, so
    an index describing content that had moved looked healthy.

    Found in production after the v6 dedup deleted rows out from under the
    index: both counts read 86 while the index was offset by one, so
    `MATCH 'Horthy'` returned a different summary and `MATCH 'IPython'`
    returned that one — the two were swapped, and search had been answering
    with the wrong records.
    """
    import sqlite3
    import sys as _sys
    import tempfile

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from summaries import SummariesBackend

    tmp = tempfile.mkdtemp(prefix="hlm-t343-")
    db = os.path.join(tmp, "digests.db")
    try:
        sb = SummariesBackend(db, os.path.join(tmp, "files"))
        for i, (title, word) in enumerate((("Alpha Report", "zebracorn"),
                                           ("Beta Report", "quixotrope"))):
            sb.add(source_url=f"https://example.invalid/t343-{i}", source_type="web",
                   title=f"{title} {word}", full_text=f"body {word}", profile_name="t343")
        sb.close()

        def match(conn, word):
            return [r[0] for r in conn.execute(
                "SELECT s.title FROM summaries_fts f JOIN summaries s ON f.rowid=s.rowid "
                "WHERE summaries_fts MATCH ?", (word,))]

        c = sqlite3.connect(db)
        assert match(c, "zebracorn") == ["Alpha Report zebracorn"], match(c, "zebracorn")

        # Corrupt the index the way a direct DELETE does: drop the index entries
        # and re-add them against the wrong rowids, leaving the counts equal.
        rows = c.execute("SELECT rowid, uuid, highlights, title, tags, full_text "
                         "FROM summaries ORDER BY rowid").fetchall()
        for r in rows:
            c.execute("INSERT INTO summaries_fts(summaries_fts, rowid, uuid, highlights, "
                      "title, tags, full_text) VALUES('delete', ?, ?, ?, ?, ?, ?)", r)
        swapped = [rows[1], rows[0]]
        for target, src in zip(rows, swapped):
            c.execute("INSERT INTO summaries_fts(rowid, uuid, highlights, title, tags, full_text) "
                      "VALUES (?, ?, ?, ?, ?, ?)", (target[0],) + src[1:])
        c.commit()
        assert c.execute("SELECT COUNT(*) FROM summaries_fts").fetchone()[0] == \
            c.execute("SELECT COUNT(*) FROM summaries").fetchone()[0], \
            "the control must leave the counts equal, or it proves nothing"
        assert match(c, "zebracorn") == ["Beta Report quixotrope"], \
            f"control did not misalign the index: {match(c, 'zebracorn')}"
        c.close()

        # Re-opening must notice and repair it.
        sb2 = SummariesBackend(db, os.path.join(tmp, "files"))
        conn = sb2._conn
        assert match(conn, "zebracorn") == ["Alpha Report zebracorn"], \
            f"misalignment survived re-init: {match(conn, 'zebracorn')}"
        assert match(conn, "quixotrope") == ["Beta Report quixotrope"], \
            f"misalignment survived re-init: {match(conn, 'quixotrope')}"
        sb2.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t347():
    """The .md records the source that was actually supplied, not only the
    canonical form.

    The canonical URL is the summary's identity — it carries the UNIQUE index
    and decides `already_exists` — but it is frequently not what the source
    *is*: `youtu.be/X` becomes `youtube.com/watch?v=X`, tracking parameters are
    dropped, a bare host gains a scheme. Writing only the transformed value
    means the file cannot say where the summary came from, and when
    canonicalization was broken it recorded `":///"` with no way back. Two real
    summaries lost their provenance that way and had to be reconstructed from
    session logs.

    The original is kept in `metadata.source_url_original` as well as the file,
    so it survives a lost file *and* a lost row.
    """
    import json as _json
    import sys as _sys
    import tempfile

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from summaries import SummariesBackend

    tmp = tempfile.mkdtemp(prefix="hlm-t347-")
    try:
        sb = SummariesBackend(os.path.join(tmp, "d.db"), os.path.join(tmp, "files"))

        def md_of(uid):
            path = sb._conn.execute(
                "SELECT summary_path FROM summaries WHERE uuid = ?", (uid,)).fetchone()[0]
            return open(os.path.join(tmp, "files", os.path.basename(path)),
                        encoding="utf-8").read()

        def uid_of(res):
            return res.get("uuid") if isinstance(res, dict) else res

        # Canonicalization changes the input → both forms are recorded.
        for raw, canonical in (
            ("https://youtu.be/abc123XYZ", "https://www.youtube.com/watch?v=abc123XYZ"),
            ("https://example.invalid/a?utm_source=x&id=7", "https://example.invalid/a?id=7"),
            ("example.invalid/bare", "https://example.invalid/bare"),
        ):
            uid = uid_of(sb.add(source_url=raw, source_type="web", title=f"T {raw[:18]}",
                                full_text="body", profile_name="t347"))
            text = md_of(uid)
            assert f"**Source:** {canonical}" in text, f"canonical missing for {raw!r}: {text[:120]}"
            assert f"**Original:** {raw}" in text, f"original missing for {raw!r}: {text[:120]}"
            meta = _json.loads(sb._conn.execute(
                "SELECT metadata FROM summaries WHERE uuid = ?", (uid,)).fetchone()[0] or "{}")
            assert meta.get("source_url_original") == raw, \
                f"original not kept in metadata for {raw!r}: {meta}"

        # Unchanged input → no redundant second line.
        for same in ("https://example.invalid/already-canonical",
                     "/home/user/notes/local-file.md"):
            uid = uid_of(sb.add(source_url=same, source_type="web", title="same",
                                full_text="body", profile_name="t347"))
            text = md_of(uid)
            assert f"**Source:** {same}" in text
            assert "**Original:**" not in text, \
                f"redundant Original line for an unchanged input {same!r}"

        # update() rewrites the file and must not drop the line.
        uid = uid_of(sb.add(source_url="https://youtu.be/keepme99", source_type="web",
                            title="before", full_text="body", profile_name="t347"))
        sb.update(uid, profile="all", title="after")
        text = md_of(uid)
        assert "**Original:** https://youtu.be/keepme99" in text, \
            f"update() dropped the original: {text[:160]}"
        assert "**Source:** https://www.youtube.com/watch?v=keepme99" in text
        sb.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t348():
    """The FTS triggers pass the content rowid, so the index tracks its content.

    `summaries_fts` is an external-content table (`content='summaries'`,
    `content_rowid='rowid'`). All three triggers omitted the rowid, and both
    halves broke:

      * INSERT without `rowid` — FTS5 assigns its own autoincrement rowid, so
        index rowids drift from content rowids and any `f.rowid = s.rowid` join
        returns the wrong record. This is the root cause of the symptom patched
        in 0.4.4, where `MATCH 'Horthy'` returned one summary and
        `MATCH 'IPython'` returned the other — the two swapped.
      * `'delete'` without `rowid` — FTS5 cannot locate the entry, so removal is
        a no-op and a stale entry is left for every row ever deleted. Measured
        on a live database: 86 content rows against 87 docsize entries.

    The user-visible consequence was that a summary added in a session was not
    findable by `search()` until the backend was reopened and the misalignment
    check rebuilt the index.

    Asserts on `summaries_fts_docsize`, not `COUNT(*) FROM summaries_fts`: the
    latter reads through to the content table and can never disagree, which is
    why the leak went unnoticed.
    """
    import sqlite3
    import sys as _sys
    import tempfile

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from summaries import SummariesBackend

    tmp = tempfile.mkdtemp(prefix="hlm-t348-")
    try:
        sb = SummariesBackend(os.path.join(tmp, "d.db"), os.path.join(tmp, "files"))
        c = sb._conn

        def counts():
            return (c.execute("SELECT COUNT(*) FROM summaries").fetchone()[0],
                    c.execute("SELECT COUNT(*) FROM summaries_fts_docsize").fetchone()[0])

        # Every trigger must name rowid, or the index silently drifts.
        for name in ("summaries_ai", "summaries_au", "summaries_ad"):
            sql = c.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()[0]
            assert "rowid" in sql, f"{name} does not pass the content rowid: {sql[:120]}"

        uid = None
        for i in range(3):
            res = sb.add(source_url=f"https://example.invalid/t348-{i}", source_type="web",
                         title=f"Zebracorn{i} Report", full_text=f"body zebracorn{i}",
                         profile_name="t348")
            uid = res.get("uuid") if isinstance(res, dict) else res
        assert counts()[0] == counts()[1], f"index leaked on insert: {counts()}"

        # The index entry must live at the *content* rowid, or joins mismatch.
        rid = c.execute("SELECT rowid FROM summaries WHERE uuid=?", (uid,)).fetchone()[0]
        assert c.execute("SELECT COUNT(*) FROM summaries_fts WHERE summaries_fts MATCH ? "
                         "AND rowid = ?", ("Zebracorn2", rid)).fetchone()[0] == 1, \
            "index rowid does not match the content rowid"

        # Each row is findable by its own distinctive term — not another's.
        for i in range(3):
            hits = c.execute("SELECT s.title FROM summaries_fts f JOIN summaries s "
                             "ON f.rowid = s.rowid WHERE summaries_fts MATCH ?",
                             (f"Zebracorn{i}",)).fetchall()
            assert [h[0] for h in hits] == [f"Zebracorn{i} Report"], \
                f"Zebracorn{i} matched the wrong record: {hits}"

        # Searchable in the same session — the user-visible failure.
        assert len(sb.search("Zebracorn2", limit=5, profile="all")) == 1, \
            "a freshly added summary is not searchable until the backend reopens"

        # Delete must remove the entry, not leak it.
        sb.delete(uid, profile_name="t348", profile="own")
        assert counts()[0] == counts()[1], f"delete leaked an index entry: {counts()}"

        # And an update must not leave the old terms behind.
        res = sb.add(source_url="https://example.invalid/t348-upd", source_type="web",
                     title="Quixotrope Before", full_text="body", profile_name="t348")
        uid2 = res.get("uuid") if isinstance(res, dict) else res
        sb.update(uid2, profile="all", title="Quixotrope After")
        assert counts()[0] == counts()[1], f"update leaked an index entry: {counts()}"
        stale = c.execute("SELECT COUNT(*) FROM summaries_fts WHERE summaries_fts MATCH ?",
                          ('"Quixotrope Before"',)).fetchone()[0]
        assert stale == 0, "the pre-update title is still indexed"
        sb.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t352():
    """A failed INSERT takes the .md file with it, whatever the failure was.

    add() writes the .md first and inserts second, deliberately — a crash
    between the two loses a file, not a record. But only the IntegrityError
    branch (the add() race) swept up the file it had just written. Any other
    INSERT failure — disk full, locked database, schema drift — left an
    orphaned .md that nothing can ever reach: list, get, cleanup and delete
    all start from a row, and there is no row.

    Simulated with a wedged execute() rather than a real disk error, since the
    branch under test is the handler, not the cause.
    """
    import sqlite3
    sb = _make_summaries("t352")
    try:
        before = set(os.listdir(sb._summaries_dir)) if os.path.isdir(sb._summaries_dir) else set()

        real = sb._conn

        class _WedgedConn:
            """Delegates everything, refuses the one INSERT under test.

            sqlite3.Connection.execute is read-only, and _conn is a property
            over a thread-local, so the handle itself is what gets swapped.
            """

            def execute(self, sql, *a, **kw):
                if sql.strip().upper().startswith("INSERT INTO SUMMARIES ("):
                    raise sqlite3.OperationalError("database or disk is full")
                return real.execute(sql, *a, **kw)

            def __getattr__(self, name):
                return getattr(real, name)

        sb._conn_local.conn = _WedgedConn()
        try:
            sb.add("https://example.com/t352", "web", "Doomed",
                   highlights=["h"], full_text="body that lands on disk first")
            raise AssertionError("add() swallowed a non-IntegrityError INSERT failure")
        except sqlite3.OperationalError:
            pass
        finally:
            sb._conn_local.conn = real

        after = set(os.listdir(sb._summaries_dir)) if os.path.isdir(sb._summaries_dir) else set()
        assert after == before, f"orphaned .md left behind: {sorted(after - before)}"

        # The connection is usable afterwards — the rollback released the
        # implicit transaction the failed INSERT left open, so the next write
        # on this thread does not block behind it.
        r = sb.add("https://example.com/t352-ok", "web", "Fine",
                   highlights=["h"], full_text="text")
        assert sb.get(r["uuid"], profile="all") is not None
    finally:
        sb.close(); _cleanup_db("t352"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t442():
    """update()'s highlights rewrite treats the new text as a literal, not a
    regex replacement template.

    `re.sub(pattern, replacement, content)` parses a *string* replacement for
    backreferences (`\\g<1>`, `\\1`-`\\9`). The old code built that string with
    an f-string: `rf"\\g<1>{highlights_section}\\3"` — interpolating
    caller-supplied highlight text directly into the template, so a backslash
    in the highlight decided what the rewrite did instead of being written
    verbatim. Measured both directions before the fix:

        highlights=["set C:\\Users\\me as home"]  -> re.error: bad escape \\U,
            raised out of update() with the SQLite UPDATE staged but not
            committed (this connection is thread-local and long-lived, and
            digests.db is shared by every profile — the dangling transaction
            then blocked a second connection to the same file with
            "database is locked")
        highlights=["cost is \\g<1>free"]  -> returned True; the .md on disk
            became "- cost is ## Highlights\\n\\nfree" (the captured section
            header duplicated into the bullet) while the DB row held the
            correct text — success reported, content silently wrong, no way
            to detect it short of reading the file back.

    A callable replacement is inserted literally by `re.sub` regardless of
    its content, which closes both. The write is also now wrapped so any
    failure (this one or an unrelated one) rolls back the staged UPDATE
    before re-raising, the same reasoning `add()` already applies to its
    INSERT.
    """
    sb = _make_summaries("t442")
    try:
        r = sb.add("https://example.com/t442", "web", "T442 probe",
                   highlights=["initial"], full_text="body text", profile_name="p1")
        uid = r["uuid"]
        md_path = sb._resolve_path(r["summary_path"])

        # A highlight containing a Windows path must not crash update(), and
        # must land in the .md exactly as given.
        windows_path = "set C:" + chr(92) + "Users" + chr(92) + "me as home"
        assert sb.update(uid, profile_name="p1", highlights=[windows_path])
        with open(md_path, encoding="utf-8") as f:
            content = f.read()
        assert windows_path in content, (
            f"backslash highlight was not written literally: {content!r}")
        assert not sb._conn.in_transaction, (
            "connection left mid-transaction after a highlights update")

        # A highlight that looks like a backreference must not be expanded.
        backref_highlight = "cost is " + chr(92) + "g<1>free"
        assert sb.update(uid, profile_name="p1", highlights=[backref_highlight])
        with open(md_path, encoding="utf-8") as f:
            content2 = f.read()
        assert backref_highlight in content2, (
            f"backreference highlight was not written literally: {content2!r}")
        assert "## Highlights\n\nfree" not in content2, (
            "the section header was duplicated into the bullet — "
            "the backreference was expanded")

        # A failure inside the rewrite (malformed highlights JSON) must roll
        # back rather than leave the shared digests.db connection locked for
        # every other profile.
        uid2 = sb.add("https://example.com/t442b", "web", "T442 probe 2",
                      highlights=["x"], full_text="body", profile_name="p1")["uuid"]
        try:
            sb.update(uid2, profile_name="p1", highlights="not valid json[")
            raise AssertionError("malformed highlights JSON did not raise")
        except AssertionError:
            raise
        except Exception:
            pass
        assert not sb._conn.in_transaction, (
            "a failed highlights update left the connection mid-transaction")
        import sqlite3
        conn2 = sqlite3.connect(sb._conn.execute("PRAGMA database_list").fetchone()[2],
                                timeout=1.0)
        try:
            conn2.execute("BEGIN IMMEDIATE")
            conn2.rollback()
        except sqlite3.OperationalError as e:
            raise AssertionError(
                f"a second connection to the shared digests.db was blocked "
                f"after the failed update: {e}")
        finally:
            conn2.close()
    finally:
        sb.close(); _cleanup_db("t442"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t464():
    """`search()`'s documented `sort_by` parameter ("created_at, updated_at.
    Default: rank (relevance)") must actually change the result order, not
    silently stay on relevance ranking for every value.

    Before this fix, `sort_by` appeared only in the signature and docstring
    — all three SQL branches in `search()` (snippet, rank-fallback,
    corruption-rebuild retry) hardcoded `ORDER BY rank`. An MCP caller
    sorting search results by recency silently got relevance order instead,
    with no error and no log line. `list_summaries()`'s same-named
    parameter was already implemented; only `search()` declared it without
    consuming it.
    """
    sb = _make_summaries("t464")
    try:
        rows = [
            ("2020-01-01T00:00:00+00:00", "t464 keyword oldest"),
            ("2021-01-01T00:00:00+00:00", "t464 keyword middle"),
            # Most keyword hits -> highest BM25 rank, but oldest by date.
            ("2019-01-01T00:00:00+00:00",
             "t464 keyword keyword keyword newest-ranked but actually oldest"),
        ]
        uuids = []
        for created_at, title in rows:
            r = sb.add(f"https://example.com/t464/{created_at}", "web", title,
                       highlights=[title], full_text=title, profile_name="p1")
            uuids.append(r["uuid"])
            sb._conn.execute(
                "UPDATE summaries SET created_at = ?, updated_at = ? WHERE uuid = ?",
                (created_at, created_at, r["uuid"]))
        sb._conn.commit()

        by_rank = sb.search("t464 keyword", profile="all")
        by_created = sb.search("t464 keyword", sort_by="created_at", profile="all")

        assert [r["created_at"] for r in by_rank] != [r["created_at"] for r in by_created], (
            "sort_by='created_at' produced the same order as the relevance "
            "default — sort_by is a no-op")
        created_dates = [r["created_at"] for r in by_created]
        assert created_dates == sorted(created_dates, reverse=True), (
            f"sort_by='created_at' did not sort newest-first: {created_dates}")
    finally:
        sb.close(); _cleanup_db("t464"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def test_t601():
    """`add()` must not let a non-list `highlights` compute a fabricated
    coverage_status from character count.

    `coverage_status = "full" if len(highlights or []) >= 3 else "partial"`
    ran on whatever `highlights` was, unchecked. A caller (or a direct
    backend call bypassing the plugin's own defensive JSON-string handling)
    passing a plain string got `len()` of the *string*, not a highlight
    count — a 3+ character string silently reported "full" coverage
    regardless of how many real highlights existed (zero, in this case).
    2026-08-23 review round 6 maintenance F6.
    """
    sb = _make_summaries("t601")
    try:
        r = sb.add("https://example.com/t601", "web", "T601",
                   highlights="ab", full_text="some full text here")
        row = sb._conn.execute(
            "SELECT highlights, coverage_status FROM summaries WHERE uuid=?",
            (r["uuid"],)).fetchone()
        assert row["highlights"] == "[]", (
            f"non-list highlights was not discarded: {row['highlights']!r}")
        assert row["coverage_status"] == "partial", (
            f"a 2-character string was counted as >=3 highlights: "
            f"coverage_status={row['coverage_status']!r}")
    finally:
        sb.close(); _cleanup_db("t601"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t602():
    """`update()`'s full_text transition must not leave the .md file and the
    DB row diverged when the file write fails after the DB write is staged.

    The .md write used to happen *before* the UPDATE, with no try/except —
    so a failure between them (or at commit()) left the file holding new
    content while the row still carried the old content_hash/summary_path,
    with no error connecting the two. The highlights-only branch right below
    this one already rolled back on a file-write failure; this branch just
    never matched it. Reordered so the UPDATE (uncommitted) runs first and
    the file write is wrapped in the same rollback-on-failure the sibling
    branch uses. 2026-08-23 recovered finding F-V.

    Reproduced by injecting a failure into the `UPDATE summaries SET` call
    itself, with the .md write left free to succeed. Pre-fix, the file write
    ran first, unconditionally — so it would have already landed on disk with
    new content by the time the (here, failing) UPDATE ran. Post-fix, the
    UPDATE runs first and this failure is reached before the file is ever
    touched, so a fresh partial-to-full transition should never create a file.
    """
    sb = _make_summaries("t602")
    try:
        r = sb.add("https://example.com/t602", "web", "T602 partial",
                   highlights=["one"], full_text=None)
        uuid = r["uuid"]
        before = sb._conn.execute(
            "SELECT content_hash, summary_path FROM summaries WHERE uuid=?",
            (uuid,)).fetchone()
        assert before["content_hash"] is None, "setup: record should start partial"
        assert not before["summary_path"], "setup: record should have no .md yet"

        real_conn = sb._conn  # triggers creation of the thread-local connection

        class _FailingConn:
            def execute(self, sql, *a, **k):
                if sql.strip().startswith("UPDATE summaries SET"):
                    raise sqlite3.OperationalError("t602 injected UPDATE failure")
                return real_conn.execute(sql, *a, **k)

            def __getattr__(self, name):
                return getattr(real_conn, name)

        sb._conn_local.conn = _FailingConn()
        try:
            raised = False
            try:
                sb.update(uuid, profile="all",
                         full_text="a brand new full summary body")
            except sqlite3.OperationalError:
                raised = True
            assert raised, "the injected UPDATE failure did not propagate"
        finally:
            sb._conn_local.conn = real_conn

        # The row's summary_path is still empty (partial), so the file this
        # transition would have created is named after it — reconstruct the
        # expected filename the same way update() does and confirm it does
        # not exist.
        import re as _re
        safe_title = _re.sub(r'[^\w\s-]', '', "T602 partial")[:100].strip().lower()
        safe_title = _re.sub(r'[\s_]+', '-', safe_title) or uuid[:8]
        would_be_path = os.path.join(sb._summaries_dir, f"{safe_title}-{uuid[:8]}.md")
        assert not os.path.exists(would_be_path), (
            "a .md file was created even though the DB UPDATE failed — file "
            "and row now diverge")

        after = sb._conn.execute(
            "SELECT content_hash, summary_path FROM summaries WHERE uuid=?",
            (uuid,)).fetchone()
        assert after["content_hash"] is None, (
            f"the UPDATE committed even though the .md write failed — "
            f"content_hash={after['content_hash']!r}, file and row now diverge")
        assert after["summary_path"] == before["summary_path"], (
            "summary_path was persisted despite the failed file write")
    finally:
        sb.close(); _cleanup_db("t602"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t603():
    """`update()`'s partial-to-full transition must warn when creating a new
    .md file for a row owned by a different profile.

    `summary_path` resolves against *this* instance's `_summaries_dir` —
    correct only when every profile shares `HLM_SUMMARIES_DIR`. delete() and
    sync() already refuse/warn on this shape; the partial-to-full file
    *creation* branch of update() did neither, silently. Not refused (the
    common single-directory configuration makes this harmless), but now
    loud. 2026-08-23 review round 7 inference-host maintenance F9 / round 10
    glm-5.2 maintenance F3.
    """
    sb = _make_summaries("t603")
    try:
        r = sb.add("https://example.com/t603", "web", "T603 foreign",
                   highlights=["one"], full_text=None, profile_name="t603-owner")
        uuid = r["uuid"]

        cap = _LogCapture()
        logging.getLogger("summaries").addHandler(cap)
        try:
            sb.update(uuid, profile_name="t603-other", profile="all",
                     full_text="full text from a different profile")
        finally:
            logging.getLogger("summaries").removeHandler(cap)

        warnings = [m for m in cap.records if "owned by profile" in m]
        assert warnings, (
            "no warning logged when creating a .md file for a foreign-owned "
            "row during the partial-to-full transition")
    finally:
        sb.close(); _cleanup_db("t603"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t604():
    """`delete_multiple` must distinguish a uuid that does not exist from one
    that exists but is refused for profile ownership — both used to collapse
    into `not_found`.

    `delete()`'s single-uuid bool return conflates the two by design (its own
    docstring says so); `delete_multiple` inherited that collapse for a batch
    report where the distinction matters — an audit of "why did my batch
    delete skip N rows" cannot tell "already gone" from "not yours" without
    it. 2026-08-23 review round 7 inference-host maintenance F3.
    """
    sb = _make_summaries("t604")
    try:
        mine = sb.add("https://example.com/t604-mine", "web", "T604 mine",
                      highlights=["a"], full_text="mine", profile_name="t604-a")
        theirs = sb.add("https://example.com/t604-theirs", "web", "T604 theirs",
                        highlights=["a"], full_text="theirs", profile_name="t604-b")

        result = sb.delete_multiple(
            [mine["uuid"], theirs["uuid"], "00000000000000000000000000000000"],
            profile_name="t604-a", profile="own")

        assert result["deleted"] == 1, f"expected 1 deleted, got {result}"
        assert result["not_found"] == 1, (
            f"the truly-nonexistent uuid was not counted as not_found: {result}")
        assert result.get("refused") == 1, (
            f"the foreign-owned row was not distinguished as refused: {result}")

        still_there = sb._conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE uuid=?", (theirs["uuid"],)
        ).fetchone()[0]
        assert still_there == 1, "the foreign row was deleted despite profile='own'"
    finally:
        sb.close(); _cleanup_db("t604"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t605():
    """`list_summaries`/`list_expiring` must clamp negative
    limit/offset/max_age_days instead of forwarding them straight into SQL.

    A negative SQLite `LIMIT` means "unlimited" — the same footgun
    `retrieve()` already guards against. `max_age_days` reached
    `timedelta(days=...)` with no sign check, putting the cutoff in the
    future. 2026-08-23 review round 6 maintenance F7 / round 7 inference-host
    maintenance F10.
    """
    sb = _make_summaries("t605")
    try:
        for i in range(3):
            sb.add(f"https://example.com/t605-{i}", "web", f"T605 {i}",
                  highlights=["a"], full_text=f"body {i}", profile_name="t605")

        result = sb.list_summaries(limit=-5, offset=-10, profile_name="t605", profile="own")
        assert result["limit"] > 0, f"negative limit was not clamped: {result['limit']}"
        assert result["offset"] == 0, f"negative offset was not clamped: {result['offset']}"
        assert len(result["records"]) <= result["limit"], (
            "a negative limit returned more rows than the clamped limit allows")

        expiring = sb.list_expiring(max_age_days=-100, profile_name="t605", profile="own")
        assert isinstance(expiring, list), "list_expiring raised on a negative max_age_days"
    finally:
        sb.close(); _cleanup_db("t605"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t492():
    """Deleting another profile's summary row must not resolve its `.md`
    against *this* instance's summaries directory.

    `summary_path` is stored relative to a summaries directory, and
    `_resolve_path` joins it to whichever directory this instance was built
    with. Under `profile="all"` — the cross-profile delete — that is the wrong
    directory for a row owned by a profile configured with a different
    `HLM_SUMMARIES_DIR`: the row goes and the file stays, orphaned and
    invisible, because nothing reads a file with no row and `sync()` only
    handles the opposite case.

    The row still deletes; what must not happen is a silent claim to have
    removed a file this instance cannot locate.
    """
    sb = _make_summaries("t492")
    try:
        res = sb.add(source_url="https://example.com/t492-foreign",
                     source_type="web", title="T492 Foreign",
                     highlights=["one"], full_text="[T492] body text",
                     tags=["t492"], profile_name="t492-own")
        uuid = res.get("uuid") if isinstance(res, dict) else res
        row = sb._conn.execute(
            "SELECT summary_path FROM summaries WHERE uuid=?", (uuid,)).fetchone()
        rel = row[0]
        md = sb._resolve_path(rel)
        assert os.path.exists(md), "setup: the .md was not written"

        # Mark the row as belonging to somebody else, then delete it the way a
        # cross-profile caller would.
        sb._conn.execute("UPDATE summaries SET profile_name='other-profile' WHERE uuid=?",
                         (uuid,))
        sb._conn.commit()
        sb.delete(uuid, profile_name="t492-own", profile="all")

        gone = sb._conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE uuid=?", (uuid,)).fetchone()[0]
        assert gone == 0, "the row was not deleted"
        assert os.path.exists(md), (
            "a foreign profile's .md was resolved against this instance's "
            "summaries directory and removed — that path is only correct when "
            "both profiles share HLM_SUMMARIES_DIR")
    finally:
        sb.close(); _cleanup_db("t492")
        shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t663():
    """Every summaries door must refuse a scope word it does not know.

    The vocabulary was enforced on exactly one of nine backend doors: `sync`
    refused an unknown word (2026-08-22 ox-alpha F1), and the other eight —
    everything reached through `_profile_sql` or `_row_profile_ok` — treated
    anything that is not the literal "all" as if the caller had said "own".
    Driven on the pre-fix tree, two profiles' rows in one store:

        list(profile="all") -> 2 records
        list(profile="al")  -> 1 record   (silently own; no error, no log)
        sync(profile="al")  -> {"error": "profile must be 'own' or 'all' ..."}

    The direction is safe — a junk word can only narrow to the caller's own
    rows, never widen — which is exactly why it survived review: the caller
    who meant "all" and typed "al" is told nothing and reads a short answer as
    a complete one. The plugin front end has no check at all; the MCP door has
    one line of it. So the guard goes where all nine doors already cross,
    and `PROFILE_SCOPES` is one definition rather than a tuple literal
    repeated per door. 2026-09-14 review round 1, bundle05 (F1).
    """
    import summaries as _summaries
    sb = _make_summaries("t663")
    try:
        sb.add("https://example.com/t663-a", "web", "Alpha",
               highlights=["h"], full_text="body a", profile_name="p-one")
        sb.add("https://example.com/t663-b", "web", "Beta",
               highlights=["h"], full_text="body b", profile_name="p-two")

        assert sb.list_summaries(profile_name="p-one", profile="own")["total"] == 1
        assert sb.list_summaries(profile_name="p-one", profile="all")["total"] == 2

        for junk in ("al", "Own", "", "hlm-test", 123, None):
            res = sb.list_summaries(profile_name="p-one", profile=junk, limit=20)
            assert res["total"] == 0 and not res["records"], (
                "list accepted the unknown scope %r and answered as if the "
                "caller had said 'own': %r" % (junk, res))
            assert sb.get("whatever", profile_name="p-one", profile=junk) is None
            assert sb.list_expiring(max_age_days=0, profile_name="p-one",
                                    profile=junk) == []

        # The row-level writers cross the other helper.
        row = sb.list_summaries(profile_name="p-one", profile="own")["records"][0]
        assert sb.delete(row["uuid"], profile_name="p-one", profile="al") is False, (
            "delete accepted an unknown scope word")
        assert sb.update(row["uuid"], profile_name="p-one", profile="al",
                         title="x") is False, (
            "update accepted an unknown scope word")
        # …and the one door that always checked still does.
        assert sb.sync(profile_name="p-one", profile="al").get("error")

        # One definition, so a third word cannot be added to one door only.
        assert _summaries.PROFILE_SCOPES == frozenset({"own", "all"})
    finally:
        sb.close(); _cleanup_db("t663")
        shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t664():
    """A batch-delete element that is not a uuid string is `invalid`, not `not_found`.

    Both front ends validated the `uuids` container and neither validated its
    elements, so `uuids=[123, "nope"]` reached SQLite as a bound integer,
    compared unequal to the TEXT `uuid` column, and came back
    `{"deleted": 0, "not_found": 2, "refused": 0}` — a type error wearing the
    response shape of two uuids that simply do not exist. An LLM that
    serialised its uuids as numbers is told its rows are gone.

    An element SQLite cannot bind at all was worse and was not in the finding:
    running this test against the pre-fix tree, a list element raised
    `sqlite3.ProgrammingError: Error binding parameter 1: type 'list' is not
    supported` out of `delete_multiple`, so the whole batch aborted and the
    real uuid beside it was never deleted. The same door therefore had two
    behaviours for one mistake — a silent `not_found` for an int, a traceback
    for a list — and the caller could act on neither.

    Counted rather than raised, because a batch call's contract is per-element
    outcomes: one bad element must not refuse the good ones beside it. In
    `delete_multiple` rather than on the two doors because they share no code
    and this is the single function both cross.
    2026-09-14 review round 1, bundle05 (F3).
    """
    sb = _make_summaries("t664")
    try:
        r = sb.add("https://example.com/t664", "web", "Gamma",
                   highlights=["h"], full_text="body", profile_name="p-one")
        res = sb.delete_multiple([123, None, ["x"], "", "no-such-uuid", r["uuid"]],
                                 profile_name="p-one", profile="own")
        assert res["invalid"] == 4, (
            "expected 4 invalid elements (int, None, list, empty string), got %r"
            % (res,))
        assert res["not_found"] == 1, (
            "the one genuinely absent uuid must still count as not_found: %r" % (res,))
        assert res["deleted"] == 1, (
            "a real uuid beside the bad ones must still be deleted: %r" % (res,))
    finally:
        sb.close(); _cleanup_db("t664")
        shutil.rmtree(sb._summaries_dir, ignore_errors=True)
