"""SQLite job history with lightweight versioning and one-up support."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


def job_key_for_url(url: str) -> str:
    h = hashlib.sha1((url or "").strip().encode("utf-8", errors="ignore")).hexdigest()
    return h[:12]


class HistoryStore:
    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._path))
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  job_key TEXT PRIMARY KEY,
                  url TEXT NOT NULL,
                  job_title TEXT,
                  job_description TEXT,
                  created_at REAL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS generations (
                  job_key TEXT NOT NULL,
                  version INTEGER NOT NULL,
                  run_id TEXT NOT NULL,
                  created_at REAL,
                  focus TEXT,
                  pipeline_preset TEXT,
                  model_sequence_json TEXT,
                  resume_md TEXT,
                  cover_md TEXT,
                  PRIMARY KEY(job_key, version)
                )
                """
            )
            con.commit()
        finally:
            con.close()

    def upsert_job(self, *, job_key: str, url: str, job_title: str | None, job_description: str | None) -> None:
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO jobs(job_key, url, job_title, job_description, created_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(job_key) DO UPDATE SET
                  url=excluded.url,
                  job_title=COALESCE(excluded.job_title, jobs.job_title),
                  job_description=COALESCE(excluded.job_description, jobs.job_description)
                """,
                (job_key, url, job_title, job_description, time.time()),
            )
            con.commit()
        finally:
            con.close()

    def next_version(self, job_key: str) -> int:
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute("SELECT COALESCE(MAX(version), 0) AS v FROM generations WHERE job_key = ?", (job_key,))
            row = cur.fetchone()
            v = int(row["v"] if row else 0)
            return v + 1
        finally:
            con.close()

    def add_generation(
        self,
        *,
        job_key: str,
        run_id: str,
        focus: str | None,
        pipeline_preset: str | None,
        model_sequence_json: str | None,
        resume_md: str | None,
        cover_md: str | None,
    ) -> int:
        version = self.next_version(job_key)
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO generations(job_key, version, run_id, created_at, focus, pipeline_preset, model_sequence_json, resume_md, cover_md)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (job_key, version, run_id, time.time(), focus, pipeline_preset, model_sequence_json, resume_md, cover_md),
            )
            con.commit()
            return version
        finally:
            con.close()

    def list_jobs(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        con = self._connect()
        try:
            cur = con.cursor()
            if limit and limit > 0:
                cur.execute(
                    """
                    SELECT j.job_key, j.url, j.job_title, j.created_at,
                           (SELECT MAX(version) FROM generations g WHERE g.job_key = j.job_key) AS latest_version
                    FROM jobs j
                    ORDER BY j.created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
            else:
                cur.execute(
                    """
                    SELECT j.job_key, j.url, j.job_title, j.created_at,
                           (SELECT MAX(version) FROM generations g WHERE g.job_key = j.job_key) AS latest_version
                    FROM jobs j
                    ORDER BY j.created_at DESC
                    """
                )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()

    def count_jobs(self) -> int:
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) AS n FROM jobs")
            row = cur.fetchone()
            return int(row["n"]) if row else 0
        finally:
            con.close()

    def delete_jobs(self, job_keys: list[str]) -> int:
        if not job_keys:
            return 0
        con = self._connect()
        try:
            cur = con.cursor()
            placeholders = ",".join("?" * len(job_keys))
            cur.execute(f"DELETE FROM generations WHERE job_key IN ({placeholders})", job_keys)
            cur.execute(f"DELETE FROM jobs WHERE job_key IN ({placeholders})", job_keys)
            con.commit()
            return cur.rowcount
        finally:
            con.close()

    def get_job(self, job_key: str) -> Optional[dict[str, Any]]:
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute("SELECT * FROM jobs WHERE job_key = ?", (job_key,))
            j = cur.fetchone()
            if not j:
                return None
            cur.execute(
                "SELECT version, run_id, created_at, focus, pipeline_preset FROM generations WHERE job_key = ? ORDER BY version DESC",
                (job_key,),
            )
            gens = [dict(r) for r in cur.fetchall()]
            out = dict(j)
            out["generations"] = gens
            return out
        finally:
            con.close()

    def get_latest_generation_content(self, job_key: str) -> Optional[dict[str, Any]]:
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM generations WHERE job_key = ? ORDER BY version DESC LIMIT 1",
                (job_key,),
            )
            r = cur.fetchone()
            return dict(r) if r else None
        finally:
            con.close()

    def get_generation_by_version(self, job_key: str, version: int) -> Optional[dict[str, Any]]:
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM generations WHERE job_key = ? AND version = ?",
                (job_key, version),
            )
            r = cur.fetchone()
            return dict(r) if r else None
        finally:
            con.close()

    def list_generations(self, job_key: str) -> list[dict[str, Any]]:
        """Return all generation records for a job, ordered newest first. Does not include resume_md/cover_md."""
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT version, run_id, created_at, focus, pipeline_preset, model_sequence_json "
                "FROM generations WHERE job_key = ? ORDER BY version DESC",
                (job_key,),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            con.close()

