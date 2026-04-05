import sqlite3
import threading
import numpy as np
from datetime import datetime


class FaceDatabase:
    def __init__(self, db_path="faces_db.sqlite"):
        self.db_path = db_path
        self._lock   = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10,
                               check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    encoding BLOB NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    x INTEGER, y INTEGER, w INTEGER, h INTEGER,
                    FOREIGN KEY (face_id) REFERENCES faces(id)
                )
            """)
            conn.commit()

    def save_face(self, name: str, encoding: np.ndarray) -> int:
        blob = np.array(encoding, dtype=np.float64).tobytes()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO faces (name, encoding, created_at) VALUES (?, ?, ?)",
                (name, blob, datetime.now().isoformat())
            )
            conn.commit()
            return cur.lastrowid

    def load_all_faces(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, encoding FROM faces"
            ).fetchall()
        return [
            {"id": r[0], "name": r[1],
             "encoding": np.array(np.frombuffer(r[2], dtype=np.float64))}
            for r in rows
        ]

    def log_sighting(self, face_id: int, bbox: tuple):
        top, right, bottom, left = bbox
        x, y, w, h = left, top, right - left, bottom - top
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO sightings (face_id, timestamp, x, y, w, h) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (face_id, datetime.now().isoformat(), x, y, w, h)
                )
                conn.commit()
        except Exception:
            pass

    def get_face_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def delete_face(self, face_id: int):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sightings WHERE face_id = ?", (face_id,))
            conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
            conn.commit()

    def list_faces(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT f.id, f.name, f.created_at, COUNT(s.id) "
                "FROM faces f LEFT JOIN sightings s ON f.id = s.face_id "
                "GROUP BY f.id ORDER BY f.created_at DESC"
            ).fetchall()
        return [{"id": r[0], "name": r[1],
                 "created_at": r[2], "sightings": r[3]} for r in rows]
