import sqlite3
from datetime import datetime

DB_PATH = "hallucination_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS claims_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            context TEXT,
            question TEXT,
            answer TEXT,
            claim TEXT,
            nli_label TEXT,
            nli_score REAL,
            human_label TEXT,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_claim(context, question, answer, claim, nli_label, nli_score):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO claims_log (timestamp, context, question, answer, claim, nli_label, nli_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), context, question, answer, claim, nli_label, nli_score))
    conn.commit()
    conn.close()