from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(
    os.environ.get(
        "PRISMOFFLINE_DATA_DIR",
        str(Path.home() / "AppData" / "Local" / "PrismOffline"),
    )
).expanduser()
DB_PATH = DATA_DIR / "prismoffline.db"

PASSWORD_ITERATIONS = 310_000
SESSION_DAYS = 30


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    with _connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                save_by_default INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS saved_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                question TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user
                ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_saved_items_user
                ON saved_items(user_id, created_at DESC);
            """
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return digest.hex()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _validate_credentials(username: str, password: str) -> tuple[str, str]:
    username = username.strip()
    if not 3 <= len(username) <= 32:
        raise ValueError("Username must be between 3 and 32 characters.")
    if not all(char.isalnum() or char in "._-" for char in username):
        raise ValueError("Username may contain letters, numbers, dot, underscore, and hyphen.")
    if len(password) < 6:
        raise ValueError("Password must contain at least 6 characters.")
    return username, password


def create_user(username: str, password: str) -> dict[str, Any]:
    username, password = _validate_credentials(username, password)
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)
    created_at = _iso(_now())

    with _connect() as db:
        try:
            cursor = db.execute(
                """
                INSERT INTO users(username, password_hash, password_salt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, password_hash, salt.hex(), created_at),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("That username is already in use.") from exc

        user_id = int(cursor.lastrowid)
        db.execute(
            "INSERT INTO user_settings(user_id, save_by_default) VALUES (?, 0)",
            (user_id,),
        )

    return {"id": user_id, "username": username, "created_at": created_at}


def authenticate(username: str, password: str) -> dict[str, Any]:
    username = username.strip()
    with _connect() as db:
        row = db.execute(
            "SELECT id, username, password_hash, password_salt FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()

    if row is None:
        raise ValueError("Invalid username or password.")

    expected = _hash_password(password, bytes.fromhex(row["password_salt"]))
    if not hmac.compare_digest(expected, row["password_hash"]):
        raise ValueError("Invalid username or password.")

    return {"id": int(row["id"]), "username": row["username"]}


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = _now()
    expires = now + timedelta(days=SESSION_DAYS)

    with _connect() as db:
        db.execute(
            """
            INSERT INTO sessions(token_hash, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (_hash_token(token), user_id, _iso(now), _iso(expires)),
        )

    return token


def get_user_for_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None

    token_hash = _hash_token(token)
    now = _iso(_now())

    with _connect() as db:
        row = db.execute(
            """
            SELECT u.id, u.username
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.expires_at > ?
            """,
            (token_hash, now),
        ).fetchone()

        db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))

    if row is None:
        return None

    return {"id": int(row["id"]), "username": row["username"]}


def delete_session(token: str | None) -> None:
    if not token:
        return
    with _connect() as db:
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))


def get_settings(user_id: int) -> dict[str, Any]:
    with _connect() as db:
        row = db.execute(
            "SELECT save_by_default FROM user_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if row is None:
            db.execute(
                "INSERT INTO user_settings(user_id, save_by_default) VALUES (?, 0)",
                (user_id,),
            )
            return {"save_by_default": False}

    return {"save_by_default": bool(row["save_by_default"])}


def update_settings(user_id: int, *, save_by_default: bool) -> dict[str, Any]:
    with _connect() as db:
        db.execute(
            """
            INSERT INTO user_settings(user_id, save_by_default)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                save_by_default = excluded.save_by_default
            """,
            (user_id, int(save_by_default)),
        )
    return {"save_by_default": bool(save_by_default)}


def save_item(
    user_id: int,
    *,
    kind: str,
    title: str,
    question: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    import json

    allowed_kinds = {"solution", "guided_attempt"}
    if kind not in allowed_kinds:
        raise ValueError("Unsupported saved item type.")

    title = title.strip() or question.strip()[:120] or "Saved question"
    question = question.strip()
    if not question:
        raise ValueError("A saved item must contain the original question.")

    created_at = _iso(_now())
    with _connect() as db:
        cursor = db.execute(
            """
            INSERT INTO saved_items(user_id, kind, title, question, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, kind, title, question, json.dumps(payload, ensure_ascii=False), created_at),
        )
        item_id = int(cursor.lastrowid)

    return {
        "id": item_id,
        "kind": kind,
        "title": title,
        "question": question,
        "payload": payload,
        "created_at": created_at,
    }


def list_items(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    import json

    limit = max(1, min(int(limit), 250))
    with _connect() as db:
        rows = db.execute(
            """
            SELECT id, kind, title, question, payload_json, created_at
            FROM saved_items
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "kind": row["kind"],
            "title": row["title"],
            "question": row["question"],
            "payload": json.loads(row["payload_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def delete_item(user_id: int, item_id: int) -> bool:
    with _connect() as db:
        cursor = db.execute(
            "DELETE FROM saved_items WHERE id = ? AND user_id = ?",
            (item_id, user_id),
        )
    return cursor.rowcount > 0
