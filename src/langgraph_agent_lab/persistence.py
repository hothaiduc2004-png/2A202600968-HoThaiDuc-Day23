"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer."""
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError("Install: pip install langgraph-checkpoint-sqlite") from exc

        db_path = database_url or "outputs/checkpoints.db"
        import os
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(conn=conn)

    if kind == "postgres":
        raise NotImplementedError("TODO: implement Postgres checkpointer (optional extension)")

    raise ValueError(f"Unknown checkpointer kind: {kind}")
