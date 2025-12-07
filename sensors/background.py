# sensors/background.py
import time
import threading
from collections import deque
from typing import Callable, Dict, Any, Optional
from .can_ingest import CANIngestWorker, CANStore
# -----------------------------
# Estado global do módulo
# -----------------------------
_mode = "sim"  # "sim" | "real"
_controls = {"throttle": 0.0, "brake": 0.0, "steer": 0.0}  # 0..1, 0..1, -1..1
_state = {
    "enabled": False,
    "rpm": 900.0,
    "speed": 0.0,     # km/h
    "coolant": 88.0,  # °C
    "steer": 0.0,     # -1..1
    "throttle": 0.0,
    "brake": 0.0,
    "timestamp": int(time.time() * 1000),
}
_history = deque(maxlen=1200)  # ~2 min @ 10 Hz

_thread: Optional[threading.Thread] = None
_stop_evt = threading.Event()
_hz = 10.0
_real_driver: Optional[Callable[[], Dict[str, Any]]] = None
_lock = threading.RLock()
_can_worker = None
_can_store = None

# -----------------------------
# Utilidades
# -----------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _sim_step(dt: float):
    with _lock:
        th = _controls["throttle"]
        br = _controls["brake"]
        st = _controls["steer"]

        spd = _state["speed"]
        clt = _state["coolant"]
        rpm = _state["rpm"]

        accel = (th * 12.0) - (br * 18.0) - (0.02 * spd)
        spd = _clamp(spd + accel * dt, 0.0, 220.0)

        base_rpm = 700.0 + spd * 45.0
        target_rpm = base_rpm + th * 1500.0
        rpm += (target_rpm - rpm) * min(1.0, dt * 3.0)
        rpm = _clamp(rpm, 650.0, 6500.0)

        clt += (th * 0.25 - 0.05) * dt
        clt = _clamp(clt, 70.0, 115.0)

        _state.update({
            "rpm": rpm,
            "speed": spd,
            "coolant": clt,
            "steer": st,
            "throttle": th,
            "brake": br,
            "timestamp": int(time.time() * 1000),
            "enabled": True,
        })

def _real_step():
    global _real_driver
    with _lock:
        if _real_driver is None:
            _state["enabled"] = False
            return
    try:
        r = _real_driver() or {}
        def pct(x, lo=0.0, hi=1.0):
            if x is None: return 0.0
            x = float(x)
            return _clamp(x/100.0 if abs(x) > 1 else x, lo, hi)

        with _lock:
            speed = float(r.get("speed") or r.get("velocidade") or _state["speed"] or 0.0)
            rpm   = float(r.get("rpm")   or _state["rpm"]   or 900.0)
            clt   = float(r.get("coolant") or r.get("temp") or _state["coolant"] or 88.0)
            th    = pct(r.get("throttle", r.get("acelerador")))
            br    = pct(r.get("brake", r.get("freio")))
            straw = r.get("steer", r.get("volante"))
            st    = _clamp(float(straw)/100.0 if straw not in (None, "") and abs(float(straw))>1 else float(straw or 0.0), -1.0, 1.0)

            _state.update({
                "rpm": rpm, "speed": speed, "coolant": clt,
                "steer": st, "throttle": th, "brake": br,
                "timestamp": int(time.time() * 1000),
                "enabled": True,
            })
    except Exception:
        with _lock:
            _state["enabled"] = False

def _loop():
    period = 1.0 / max(1.0, _hz)
    last = time.time()
    while not _stop_evt.is_set():
        now = time.time()
        dt = now - last
        last = now

        if get_mode() == "sim":
            step = 0.05
            t = dt
            while t > 0:
                _sim_step(min(step, t))
                t -= step
        else:
            _real_step()

        reading = {
            "t": _state["timestamp"],
            "rpm": round(_state["rpm"], 0),
            "speed": round(_state["speed"], 2),
            "coolant": round(_state["coolant"], 1),
            "steer": round(_state["steer"], 3),
            "throttle": round(_state["throttle"], 3),
            "brake": round(_state["brake"], 3),
            "mode": _mode,
        }

        with _lock:
            _history.append(reading) # !!!

        rem = period - (time.time() - now)
        if rem > 0:
            _stop_evt.wait(rem)

# -----------------------------
# API pública
# -----------------------------
def start_background(mode: str = None, hz: float = 10.0,
                     real_driver: Optional[Callable[[], Dict[str, Any]]] = None) -> None:
    global _thread, _hz, _mode, _real_driver, _can_worker, _can_store
    _mode = mode if mode in ("sim", "real") else "sim"
    _hz = max(1.0, float(hz))
    _real_driver = real_driver

    if (_mode == "real") :
        if _can_store is None:
            _can_store = CANStore()
        if _can_worker is None or not _can_worker.enabled:
            _can_worker = CANIngestWorker(_can_store, enabled=True)
            _can_worker.start()

    else:
        if _thread and _thread.is_alive():
            return
        _stop_evt.clear()
        with _lock:
            _state["enabled"] = (_mode == "sim")
        _thread = threading.Thread(target=_loop, name="sensors-bg", daemon=True)
        _thread.start()

def stop_background() -> None:
    global _thread
    if _thread and _thread.is_alive():
        _stop_evt.set()
        _thread.join(timeout=2.0)
    with _lock:
        _state["enabled"] = False
    _thread = None

def is_running() -> bool:
    t = _thread
    return bool(t and t.is_alive())

def set_mode(mode: str) -> None:
    global _mode
    if mode in ("sim", "real"):
        _mode = mode

def get_mode() -> str:
    return _mode

def set_controls(**payload) -> Dict[str, float]:
    """
    Aceita:
      - throttle/brake/steer  (0..1, -1..1)
      - throttlePct/brakePct  (0..100)
      - steerPct              (-100..100)
    """
    with _lock:
        if "throttlePct" in payload:
            _controls["throttle"] = _clamp(float(payload["throttlePct"]) / 100.0, 0.0, 1.0)
        if "brakePct" in payload:
            _controls["brake"] = _clamp(float(payload["brakePct"]) / 100.0, 0.0, 1.0)
        if "steerPct" in payload:
            _controls["steer"] = _clamp(float(payload["steerPct"]) / 100.0, -1.0, 1.0)

        if "throttle" in payload:
            _controls["throttle"] = _clamp(float(payload["throttle"]), 0.0, 1.0)
        if "brake" in payload:
            _controls["brake"] = _clamp(float(payload["brake"]), 0.0, 1.0)
        if "steer" in payload:
            val = float(payload["steer"])
            if abs(val) > 1.0:  # se vier -100..100
                val = val / 100.0
            _controls["steer"] = _clamp(val, -1.0, 1.0)

        return dict(_controls)

def get_latest() -> Dict[str, Any]:
    if (_mode == "real") and _can_store is not None:
        readings = _can_store.get_readings(50)
        return readings
    else:
        with _lock:
            return {
                "graph": dict(_state),
                "class": None              
            }

def get_history(n: int = 120) -> Dict[str, Any]:
    with _lock:
        maxlen = getattr(_history, "maxlen", None) or len(_history) or 1
        n = max(1, min(int(n or 1), maxlen))
        data = list(_history)[-n:]
        return {"ok": True, "data": data}
