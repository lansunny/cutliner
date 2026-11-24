"""Persistent Cutliner catalog. Media and execution evidence stay on disk."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time


SCHEMA_VERSION = 1
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS studio_schema (version INTEGER PRIMARY KEY)",
    """CREATE TABLE IF NOT EXISTS studio_media (
          id TEXT PRIMARY KEY, path TEXT NOT NULL, role TEXT NOT NULL CHECK (role IN ('source','artifact')),
          favorite INTEGER NOT NULL CHECK(favorite IN (0,1)),
          updated_at DOUBLE PRECISION NOT NULL, payload TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS studio_runs (
          id TEXT PRIMARY KEY, media_id TEXT NOT NULL, path TEXT NOT NULL,
          updated_at DOUBLE PRECISION NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS studio_jobs (
          id TEXT PRIMARY KEY, status TEXT NOT NULL, started DOUBLE PRECISION NOT NULL, payload TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS studio_media_order ON studio_media(role, favorite, updated_at)",
    "CREATE INDEX IF NOT EXISTS studio_runs_media ON studio_runs(media_id, updated_at)",
    "CREATE INDEX IF NOT EXISTS studio_jobs_order ON studio_jobs(started)",
)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class StudioStore:
    def __init__(self, sqlite_path: Path, database_url: str = ""):
        self.path = Path(sqlite_path)
        self._dsn = database_url.strip()
        self.backend = "postgresql" if self._dsn else "sqlite"

    @contextmanager
    def connection(self):
        if self.backend == "postgresql":
            import psycopg
            connection = psycopg.connect(self._dsn, connect_timeout=5, application_name="cutliner")
        else:
            connection = sqlite3.connect(self.path, timeout=10)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def execute(self, connection, sql, parameters=()):
        return connection.execute(sql.replace("?", "%s") if self.backend == "postgresql" else sql, parameters)

    def initialize(self):
        if self.backend == "sqlite":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            if self.backend == "sqlite":
                connection.execute("PRAGMA journal_mode = WAL")
            else:
                # Prevent concurrent first launches from racing on schema creation.
                connection.execute("SELECT pg_advisory_xact_lock(308118661)")
            connection.execute(SCHEMA[0])
            version = connection.execute("SELECT MAX(version) FROM studio_schema").fetchone()[0]
            if version is not None and version > SCHEMA_VERSION:
                raise ValueError("Database schema is newer than this Cutliner version")
            if version != SCHEMA_VERSION:
                for statement in SCHEMA[1:]:
                    connection.execute(statement)
                self.execute(connection, "INSERT INTO studio_schema (version) VALUES (?) ON CONFLICT DO NOTHING",
                             (SCHEMA_VERSION,))

    def save_media(self, record):
        with self.connection() as connection:
            self.execute(connection, """INSERT INTO studio_media (id,path,role,favorite,updated_at,payload)
                VALUES (?,?,?,?,?,?) ON CONFLICT (id) DO UPDATE SET path=excluded.path, role=excluded.role,
                favorite=excluded.favorite, updated_at=excluded.updated_at,payload=excluded.payload""",
                (record["id"], record["path"], record["role"], int(record["favorite"]), record["updated_at"], encode(record)))

    def media(self):
        with self.connection() as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT payload FROM studio_media ORDER BY favorite DESC, updated_at DESC, id").fetchall()]

    def save_run(self, record):
        with self.connection() as connection:
            self.execute(connection, """INSERT INTO studio_runs (id,media_id,path,updated_at) VALUES (?,?,?,?)
                ON CONFLICT (id) DO UPDATE SET media_id=excluded.media_id,path=excluded.path,updated_at=excluded.updated_at""",
                (record["id"], record["media_id"], record["path"], record["updated_at"]))

    def run_paths(self):
        with self.connection() as connection:
            return [Path(row[0]) for row in connection.execute(
                "SELECT path FROM studio_runs ORDER BY updated_at DESC, id").fetchall()]

    def save_job(self, record, *, preserve_existing=False):
        # Source media and artifact versions are rebound from audit files at
        # startup. Do not duplicate a growing run projection in job storage.
        payload = {k: v for k, v in record.items() if k != "run"}
        if isinstance(payload.get("result"), dict):
            payload["result"] = {k: v for k, v in payload["result"].items() if k != "run"}
        conflict = ("DO NOTHING" if preserve_existing else
                    "DO UPDATE SET status=excluded.status, started=excluded.started, payload=excluded.payload")
        with self.connection() as connection:
            self.execute(connection, f"INSERT INTO studio_jobs (id,status,started,payload) VALUES (?,?,?,?) ON CONFLICT (id) {conflict}",
                         (record["id"], record["status"], record.get("started", time.time()), encode(payload)))

    def jobs(self):
        with self.connection() as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT payload FROM studio_jobs ORDER BY started, id").fetchall()]
