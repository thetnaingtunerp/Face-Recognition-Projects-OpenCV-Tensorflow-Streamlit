import sqlite3
from contextlib import contextmanager
from datetime import datetime
from config import DB_PATH, DATA_DIR

@contextmanager
def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_connection() as conn:
        conn.executescript("""
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            department TEXT DEFAULT '',
            email TEXT DEFAULT '',
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_employee
            ON face_embeddings(employee_id);
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            attendance_date TEXT NOT NULL,
            check_in TEXT NOT NULL,
            check_out TEXT,
            recognition_method TEXT NOT NULL DEFAULT 'FaceNet',
            FOREIGN KEY(employee_id) REFERENCES employees(employee_id),
            UNIQUE(employee_id, attendance_date)
        );
        CREATE INDEX IF NOT EXISTS idx_attendance_date
            ON attendance(attendance_date);
        """)

def add_employee(employee_id, full_name, department='', email=''):
    now = datetime.now().isoformat(timespec='seconds')
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO employees(employee_id, full_name, department, email, created_at) VALUES (?, ?, ?, ?, ?)",
            (employee_id.strip(), full_name.strip(), department.strip(), email.strip(), now)
        )

def get_employees(active_only=True):
    query = "SELECT * FROM employees"
    if active_only:
        query += " WHERE active=1"
    query += " ORDER BY full_name"
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(query).fetchall()]

def set_employee_active(employee_id, active):
    with get_connection() as conn:
        conn.execute("UPDATE employees SET active=? WHERE employee_id=?", (int(active), employee_id))

def add_embedding(employee_id, embedding):
    import numpy as np
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO face_embeddings(employee_id, embedding, created_at) VALUES (?, ?, ?)",
            (employee_id, arr.tobytes(), datetime.now().isoformat(timespec='seconds'))
        )

def get_all_embeddings():
    import numpy as np
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT e.employee_id, e.full_name, f.embedding
            FROM face_embeddings f
            JOIN employees e ON e.employee_id=f.employee_id
            WHERE e.active=1
            ORDER BY e.employee_id
        """).fetchall()
    result = []
    for row in rows:
        result.append({
            "employee_id": row["employee_id"],
            "full_name": row["full_name"],
            "embedding": np.frombuffer(row["embedding"], dtype=np.float32).copy()
        })
    return result

def record_check_in(employee_id, method='FaceNet'):
    today = datetime.now().date().isoformat()
    now = datetime.now().isoformat(timespec='seconds')
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM attendance WHERE employee_id=? AND attendance_date=?",
            (employee_id, today)
        ).fetchone()
        if existing:
            return False, "Already checked in today"
        conn.execute(
            "INSERT INTO attendance(employee_id, attendance_date, check_in, recognition_method) VALUES (?, ?, ?, ?)",
            (employee_id, today, now, method)
        )
    return True, "Check-in recorded"

def record_check_out(employee_id):
    today = datetime.now().date().isoformat()
    now = datetime.now().isoformat(timespec='seconds')
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE attendance SET check_out=? WHERE employee_id=? AND attendance_date=? AND check_out IS NULL",
            (now, employee_id, today)
        )
        return cur.rowcount > 0

def get_attendance(start_date=None, end_date=None):
    query = """
        SELECT a.id, a.employee_id, e.full_name, e.department,
               a.attendance_date, a.check_in, a.check_out, a.recognition_method
        FROM attendance a JOIN employees e ON e.employee_id=a.employee_id
        WHERE 1=1
    """
    params = []
    if start_date:
        query += " AND a.attendance_date >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND a.attendance_date <= ?"
        params.append(str(end_date))
    query += " ORDER BY a.attendance_date DESC, a.check_in DESC"
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]

def attendance_summary():
    today = datetime.now().date().isoformat()
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM employees WHERE active=1").fetchone()[0]
        present = conn.execute("SELECT COUNT(*) FROM attendance WHERE attendance_date=?", (today,)).fetchone()[0]
    return {"total": total, "present": present, "absent": max(total-present, 0)}
