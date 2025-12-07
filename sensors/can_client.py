# sensors/can_client.py
# Você pode usar direto o store (modo "in-process") ou HTTP (se preferir desacoplar).
import os, requests
from typing import List, Dict, Any, Optional

class CANClient:
    def __init__(self, base_url: Optional[str] = None, in_process_fn=None):
        """
        base_url: use HTTP (ex.: http://localhost:5001) para /can/*
        in_process_fn: função que retorna o store (usada para acesso direto, sem HTTP)
        """
        self.base_url = base_url
        self.in_process_fn = in_process_fn

    def readings(self, limit: int = 50) -> List[Dict[str, Any]]:
        if self.in_process_fn:
            store = self.in_process_fn()
            return store.get_readings(limit=limit)
        else:
            r = requests.get(f"{self.base_url}/can/readings", params={"limit": limit}, timeout=5)
            r.raise_for_status()
            return r.json()

    def stats(self) -> Dict[str, Any]:
        if self.in_process_fn:
            store = self.in_process_fn()
            return store.stats()
        else:
            r = requests.get(f"{self.base_url}/can/stats", timeout=5)
            r.raise_for_status()
            return r.json()
