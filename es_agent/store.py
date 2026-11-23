"""SQLite 持久化：会话、消息历史、工具调用日志、待卖家审批的工单。只用标准库 sqlite3。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, created_at REAL, updated_at REAL, history_json TEXT
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, role TEXT, content TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS tool_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, name TEXT, arguments TEXT,
  result TEXT, ms INTEGER, created_at REAL
);
CREATE TABLE IF NOT EXISTS tickets (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, kind TEXT, status TEXT,
  order_number TEXT, email TEXT, summary_zh TEXT, payload TEXT, created_at REAL, decided_at REAL
);
"""


def _encode(obj):
    if is_dataclass(obj):
        return {"__toolcall__": asdict(obj)}
    raise TypeError(type(obj))


def _decode(d):
    if "__toolcall__" in d:
        from .llm import ToolCall
        return ToolCall(**d["__toolcall__"])
    return d


class Store:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    # 会话
    def new_conversation(self) -> str:
        cid = uuid.uuid4().hex[:12]
        now = time.time()
        self._exec("INSERT INTO conversations VALUES (?,?,?,?)", (cid, now, now, "[]"))
        return cid

    def get_history(self, cid: str) -> list[dict] | None:
        row = self._exec("SELECT history_json FROM conversations WHERE id=?", (cid,)).fetchone()
        return None if row is None else json.loads(row["history_json"], object_hook=_decode)

    def save_history(self, cid: str, history: list[dict]) -> None:
        self._exec("UPDATE conversations SET history_json=?, updated_at=? WHERE id=?",
                   (json.dumps(history, default=_encode, ensure_ascii=False), time.time(), cid))

    def add_message(self, cid: str, role: str, content: str) -> None:
        self._exec("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?,?,?,?)",
                   (cid, role, content, time.time()))

    def messages(self, cid: str) -> list[dict]:
        rows = self._exec("SELECT role, content, created_at FROM messages WHERE conversation_id=? ORDER BY id",
                          (cid,)).fetchall()
        return [dict(r) for r in rows]

    def recent_conversations(self, limit: int = 20) -> list[dict]:
        rows = self._exec("""SELECT c.id, c.updated_at,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS n_messages,
               (SELECT COUNT(*) FROM tickets t WHERE t.conversation_id=c.id) AS n_tickets
               FROM conversations c ORDER BY c.updated_at DESC LIMIT ?""", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # 工具日志
    def log_tool(self, cid: str, name: str, args: dict, result: str, ms: int) -> None:
        self._exec("INSERT INTO tool_calls (conversation_id, name, arguments, result, ms, created_at) "
                   "VALUES (?,?,?,?,?,?)", (cid, name, json.dumps(args, ensure_ascii=False), result, ms, time.time()))

    # 工单
    def create_ticket(self, cid: str, kind: str, summary_zh: str, payload: dict,
                      order_number: str = "", email: str = "") -> int:
        cur = self._exec("INSERT INTO tickets (conversation_id, kind, status, order_number, email, summary_zh, "
                         "payload, created_at) VALUES (?,?,?,?,?,?,?,?)",
                         (cid, kind, "pending", order_number, email, summary_zh,
                          json.dumps(payload, ensure_ascii=False), time.time()))
        return cur.lastrowid

    def tickets(self, status: str | None = None) -> list[dict]:
        if status:
            rows = self._exec("SELECT * FROM tickets WHERE status=? ORDER BY id DESC", (status,)).fetchall()
        else:
            rows = self._exec("SELECT * FROM tickets ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    def get_ticket(self, tid: int) -> dict | None:
        r = self._exec("SELECT * FROM tickets WHERE id=?", (tid,)).fetchone()
        return dict(r) if r else None

    def decide_ticket(self, tid: int, status: str) -> None:
        self._exec("UPDATE tickets SET status=?, decided_at=? WHERE id=?", (status, time.time(), tid))

    def stats(self) -> dict:
        total = self._exec("SELECT COUNT(*) FROM conversations WHERE history_json != '[]'").fetchone()[0]
        with_ticket = self._exec("SELECT COUNT(DISTINCT conversation_id) FROM tickets").fetchone()[0]
        pending = self._exec("SELECT COUNT(*) FROM tickets WHERE status='pending'").fetchone()[0]
        return {"conversations": total, "auto_resolved": total - with_ticket, "pending_tickets": pending,
                "auto_rate": round((total - with_ticket) / total, 3) if total else 0.0}
