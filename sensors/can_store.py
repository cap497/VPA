# sensors/can_store.py
import sqlite3, json, os, threading, time, csv
from typing import List, Dict, Any, Optional
from rede.centralized import classify_input
import numpy as np
import pandas as pd

DEFAULT_DB = os.environ.get("CAN_DB_PATH", os.path.abspath("./data/can_readings.sqlite"))
DEFAULT_CSV = os.environ.get("CAN_CSV_PATH", os.path.abspath("./data/can_readings.csv"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_ms INTEGER NOT NULL,
  can_id TEXT NOT NULL,
  payload TEXT NOT NULL,
  classes TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ts ON readings(ts_ms DESC);
CREATE INDEX IF NOT EXISTS idx_can ON readings(can_id);
"""

class CANStore:
    def __init__(self, db_path: str = DEFAULT_DB, csv_path: str = DEFAULT_CSV):
        self.db_path = db_path
        self.csv_path = csv_path
        self._lock = threading.RLock()
        self._init()

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "timestamp", "speed", "rpm", "throttle_position", "acc_long", "acc_lat"])
                writer.writeheader()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init(self):
        with self._conn() as c:
            for stmt in SCHEMA.strip().split(";"):
                s = stmt.strip()
                if s: c.execute(s)

    def add_reading(self, ts_ms: int, can_id: str, payload: Dict[str, Any]):
        data = json.dumps(payload, ensure_ascii=False)
        input = {
            "speed": payload.get("speed", 0),
            "acc_long": payload.get("acc_long", 0),
            "acc_lat": payload.get("acc_lat", 0),
            "acc_norm": payload.get("acc_norm", 0),
            "engine_speed": payload.get("engine_speed", 0),
            "throttle_position": payload.get("throttle_position", 0),
            "delta_acc_lat": payload.get("jerk", 0)
        }
        # Classify using both models
        try:
            fedavg_pred = classify_input(pd.DataFrame([input], dtype=np.float32), "FedAvg")[0]
            fedprox_pred = classify_input(pd.DataFrame([input], dtype=np.float32), "FedProx")[0]
            central_pred = classify_input(pd.DataFrame([input], dtype=np.float32), "central")[0]
            

        except Exception as e:
            print(f"Erro na classificação centralizada: {e}")
            central_pred = 0.0
            fedavg_pred = 0.0
            fedprox_pred = 0.0

        classes = json.dumps([central_pred, fedavg_pred, fedprox_pred], ensure_ascii=False)
        
        # Store in DB
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO readings(ts_ms, can_id, payload, classes) VALUES(?,?,?,?)",
                      (ts_ms, can_id, data, classes))
            new_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
            
        #Store in CSV
        # TODO: put cluster
        with self._lock, open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "timestamp", "speed", "acc_norm", "engine_speed", "throttle_position", "delta_acc_lat"])
            writer.writerow({
                "id": new_id,
                "timestamp": ts_ms,
                "speed": payload.get("speed", ""),
                "acc_norm": payload.get("acc_norm", ""),
                "engine_speed": payload.get("engine_speed", ""),
                "throttle_position": payload.get("throttle_position", ""),
                "delta_acc_lat": payload.get("jerk", "")
            })

    def get_readings(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as c:
            cur = c.execute("SELECT ts_ms, can_id, payload, classes FROM readings ORDER BY ts_ms DESC LIMIT ?", (limit,))
            graph = []
            classes = []
            rows = cur.fetchall()
            for ts, cid, pl, _ in rows:
                try:
                    payload = json.loads(pl)
                except Exception:
                    payload = {"raw": pl}
                graph.append({"timestamp": ts, "id": cid, "data": payload,})
            
            if rows:
                classes = json.loads(rows[0][3])
                
            out = {
            "graph": graph,
            "classes": classes
            }
            return out

    def stats(self) -> Dict[str, Any]:
        with self._lock, self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
            last = c.execute("SELECT MAX(ts_ms) FROM readings").fetchone()[0]
            return {"total": total, "last_ts_ms": last}
