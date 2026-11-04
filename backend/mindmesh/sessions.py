"""Per-visitor private workspaces.

Each browser page load generates a random 128-bit session id and sends it with every request. The id maps to its
own PostgreSQL schema (`s_<id>`), so one visitor can never query another's documents, chunks or graph, and the id
is unguessable. A reload creates a new id, so the previous workspace is dropped: by a beacon sent when the page
is closed (best effort) and, as the guaranteed backstop, by an idle-timeout sweep.
"""
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

import psycopg
import psycopg.sql

from . import config
from .retrieval import Index
from .storage import Store

SID_RE = re.compile(r"^[a-f0-9]{32}$")
MAX_CACHED = 6          # open DB connections kept per server instance
TOUCH_EVERY = 300       # seconds between last_seen refreshes
SWEEP_EVERY = 60        # seconds between idle sweeps


def _schema(sid):
    return f"s_{sid}"


class Sessions:
    def __init__(self):
        self.lock = threading.RLock()
        self.cache = OrderedDict()   # sid -> [store, index, last_touch]
        self.last_sweep = 0.0

    # -- registry (public.mm_sessions) ------------------------------------
    def _admin(self):
        db = psycopg.connect(config.DATABASE_URL, autocommit=True)
        db.execute("CREATE TABLE IF NOT EXISTS public.mm_sessions(sid TEXT PRIMARY KEY, last_seen DOUBLE PRECISION)")
        return db

    def _touch(self, sid):
        with self._admin() as db:
            db.execute("INSERT INTO public.mm_sessions VALUES(%s,%s) "
                       "ON CONFLICT (sid) DO UPDATE SET last_seen=EXCLUDED.last_seen", (sid, time.time()))

    def _drop(self, db, sid):
        db.execute(psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(psycopg.sql.Identifier(_schema(sid))))
        db.execute("DELETE FROM public.mm_sessions WHERE sid=%s", (sid,))

    def _sweep(self):
        now = time.time()
        if now - self.last_sweep < SWEEP_EVERY:
            return
        self.last_sweep = now
        with self._admin() as db:
            old = [r[0] for r in db.execute("SELECT sid FROM public.mm_sessions WHERE last_seen < %s",
                                            (now - config.SESSION_TTL,)).fetchall()]
            for sid in old:
                self._forget(sid)
                self._drop(db, sid)

    def _forget(self, sid):
        with self.lock:
            entry = self.cache.pop(sid, None)
        if entry:
            self._remove_files(entry[0])
            try:
                entry[0].close()
            except Exception:
                pass

    @staticmethod
    def _remove_files(store):
        try:
            for d in store.list_documents():
                for p in (d.get("path"), (d.get("meta") or {}).get("annotated")):
                    if p:
                        Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    # -- public API ---------------------------------------------------------
    def get(self, sid):
        """Return (store, index) for a valid session id, creating the private schema on first use."""
        if not SID_RE.match(sid or ""):
            raise ValueError("invalid session id")
        now = time.time()
        with self.lock:
            entry = self.cache.get(sid)
            if entry:
                self.cache.move_to_end(sid)
                if now - entry[2] > TOUCH_EVERY:
                    entry[2] = now
                    self._touch(sid)
                return entry[0], entry[1]
        self._touch(sid)
        self._sweep()
        store = Store(config.DATABASE_URL, schema=_schema(sid))
        entry = [store, Index(store), now]
        with self.lock:
            self.cache[sid] = entry
            while len(self.cache) > MAX_CACHED:
                _, (old_store, _, _) = self.cache.popitem(last=False)
                try:
                    old_store.close()   # data stays in Postgres until the session ends or expires
                except Exception:
                    pass
        return entry[0], entry[1]

    def end(self, sid):
        """Delete a session's workspace immediately (page closed or reloaded)."""
        if not SID_RE.match(sid or ""):
            return
        self._forget(sid)
        with self._admin() as db:
            self._drop(db, sid)


sessions = Sessions()
