#!/usr/bin/env python3
import os, time, json as _json, inspect
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from typing import Any, Dict, List, Optional, Sequence, Mapping, Iterable
from sensors.can_store import CANStore
from random import random
from threading import Thread

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, abort
)

# ===== Sensores: import funções públicas =====
from sensors import (
    start_background as _start_bg,
    stop_background  as _stop_bg,
    is_running       as _is_running,
    get_latest       as _get_latest,
    get_history      as _get_history,
    set_controls     as _set_controls,
    set_mode         as _set_mode,
    get_mode         as _get_mode,
)

# ===== RAG núcleo =====
from query_core import (
    load_assets_index, load_bm25_index, load_faiss_index,
    run_rag_pipeline, ensure_idle_monitor
)

# -----------------------------------------------------------------------------
# App & Config
# -----------------------------------------------------------------------------
app = Flask(__name__)

EXTRACT_BASE   = os.environ.get("EXTRACT_BASE", os.path.abspath("./assets_out"))
INDEX_BASE     = os.environ.get("INDEX_BASE",   os.path.abspath("./indices"))
LM_STUDIO_BASE = os.environ.get("LM_STUDIO_BASE", "http://localhost:1234")

MODEL_MAP = {
    "fiat argo": "argo", "hyundai hb20": "hb20", "renault kwid": "kwid",
    "chevrolet onix": "onix", "vw t-cross": "tcross",
    "argo": "argo", "hb20": "hb20", "kwid": "kwid", "onix": "onix",
    "tcross": "tcross", "t-cross": "tcross",
}
def normalize_model(name: str) -> str:
    if not name: return ""
    return MODEL_MAP.get(name.strip().lower(), name.strip().lower())

# -----------------------------------------------------------------------------
# Helpers defensivos para SENSORS
# -----------------------------------------------------------------------------
def _safe_json():
    """Garante dict mesmo se vier lista/None."""
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return {}

def _call_with_supported(func, **kwargs):
    """Passa só kwargs que a função realmente aceita e que não são None."""
    sig = inspect.signature(func)
    supported = {k: v for k, v in kwargs.items() if v is not None and k in sig.parameters}
    return func(**supported)

def start_background(mode=None, source=None, mqtt_url=None, mqtt_topic=None, hz=None):
    """Wrapper compatível com várias assinaturas."""
    # Alguns drivers só aceitam (mode, hz); outros aceitam (source, mqtt_url, mqtt_topic, hz) etc.
    return _call_with_supported(
        _start_bg,
        mode=mode, source=source, mqtt_url=mqtt_url, mqtt_topic=mqtt_topic, hz=hz
    )

def stop_background():
    return _stop_bg()

def is_running():
    try:
        return bool(_is_running())
    except Exception:
        return False

def set_mode(mode):
    return _call_with_supported(_set_mode, mode=mode)

def get_mode():
    try:
        m = _get_mode()
        return (m or "sim").lower()
    except Exception:
        return "sim"

def set_controls(throttle=None, brake=None, steer=None):
    return _call_with_supported(_set_controls, throttle=throttle, brake=brake, steer=steer)

def _as_list(obj: Any) -> List[Any]:
    """Converte obj em lista de forma segura (evita strings/dicts serem iterados)."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    # evita iterar string/bytes/dict como sequência
    return [obj] if isinstance(obj, (str, bytes, dict)) else list(obj) if hasattr(obj, "__iter__") else []

def get_history(n: int = 120) -> Dict[str, Any]:
    """
    Retorna {"ok": bool, "data": List[dict]}.
    Aceita _get_history(n=...) (novo) ou _get_history() (legado).
    """
    if 'n' in inspect.signature(_get_history).parameters:
        res = _get_history(n=n)
    else:
        res = _get_history()

    data_list: List[Any] = []
    if isinstance(res, Mapping):
        raw = res.get("data", [])
        data_list = _as_list(raw)  # sua função que evita strings/bytes
    else:
        data_list = _as_list(res)

    data_dicts: List[dict] = [x for x in data_list if isinstance(x, dict)]

    if n is not None and n > 0 and len(data_dicts) > n:
        data_dicts = data_dicts[-n:]

    return {"ok": len(data_dicts) > 0, "data": data_dicts}

def _call_with_optional_n(fn, n):
    """Chama fn com n se for suportado: keyword, posicional, ou sem n."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # fallback p/ callables sem signature (C-builtins, wrappers)
        try:
            return fn(n=n)
        except TypeError:
            try:
                return fn(n)
            except TypeError:
                return fn()

    params = sig.parameters

    # Se aceita **kwargs, podemos passar n como keyword.
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(n=n)

    # Se tem um parâmetro chamado n:
    if 'n' in params:
        kind = params['n'].kind
        if kind is inspect.Parameter.POSITIONAL_ONLY:
            return fn(n)           # passa como posicional
        else:
            return fn(n=n)         # passa como keyword

    # Não tem n -> chamada legada sem argumento
    return fn()


def get_latest(n: Optional[int] = None) -> List[dict]:
    """
    Retorna sempre uma lista de dicts (mais recentes).
    Suporta _get_latest(n=...) (novo), _get_latest(n, /) (posicional) e _get_latest() (legado).
    """
    res: Any = _call_with_optional_n(_get_latest, n)

    if res is None:
        return []

    # Normalização
    if isinstance(res, Mapping):
        items = list(res.get("data", [])) if isinstance(res.get("data"), (list, tuple)) else [dict(res)]
    elif isinstance(res, Sequence) and not isinstance(res, (str, bytes)):
        items = list(res)
    elif isinstance(res, dict):
        items = [res]
    else:
        return []

    dicts = [x for x in items if isinstance(x, dict)]
    if n is not None and n > 0 and len(dicts) > n:
        dicts = dicts[-n:]
    return dicts

# Fonte REAL (usado se seu driver suportar esses kwargs)
CAN_SOURCE = os.environ.get("CAN_SOURCE", "real")
MQTT_URL   = os.environ.get("MQTT_URL", "mqtt://localhost:1883")
MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "futurelab/can")
CAN_HZ     = int(os.environ.get("CAN_HZ", "10"))
_first_real_call_done = False

# -----------------------------------------------------------------------------
# Telemetria API
# -----------------------------------------------------------------------------
@app.route("/telemetry/mode", methods=["POST"])
def telemetry_mode():
    payload = _safe_json()
    mode = (payload.get("mode") or "").lower()
    if mode not in ("sim", "real"):
        return jsonify({"ok": False, "error": "mode must be 'sim' or 'real'"}), 400
    set_mode(mode)
    if not is_running():
        if mode == "real":
            start_background(source=CAN_SOURCE, mqtt_url=MQTT_URL, mqtt_topic=MQTT_TOPIC, hz=CAN_HZ)
        else:
            start_background(mode="sim", hz=10.0)
    return jsonify({"ok": True, "mode": get_mode(), "running": is_running()})

@app.route("/telemetry/start", methods=["POST"])
def telemetry_start():
    payload = _safe_json()
    mode = (payload.get("mode") or get_mode() or "sim").lower()
    if mode == "real":
        start_background(source=CAN_SOURCE, mqtt_url=MQTT_URL, mqtt_topic=MQTT_TOPIC, hz=CAN_HZ)
    else:
        start_background(mode="sim", hz=10.0)
    return jsonify({"ok": True, "mode": get_mode(), "running": is_running()})

@app.route("/telemetry/stop", methods=["POST"])
def telemetry_stop():
    stop_background()
    return jsonify({"ok": True, "running": is_running()})

@app.route("/sim_sensors/toggle", methods=["POST"])
def sim_toggle():
    new_mode = "real" if get_mode() == "sim" else "sim"
    set_mode(new_mode)
    if new_mode == "real":
        start_background(source=CAN_SOURCE, mqtt_url=MQTT_URL, mqtt_topic=MQTT_TOPIC, hz=CAN_HZ)
    else:
        start_background(mode="sim", hz=10.0)
    return jsonify({"enabled": new_mode == "sim", "mode": new_mode})

@app.route("/sim_sensors/readings")
def sim_readings():
    global _first_real_call_done
    _first_real_call_done=False
    if not is_running():
        if get_mode() == "real":
            start_background(source=CAN_SOURCE, mqtt_url=MQTT_URL, mqtt_topic=MQTT_TOPIC, hz=CAN_HZ)
        else:
            start_background(mode="sim", hz=10.0)
        time.sleep(0.05)

    rows = get_latest(n=1)
    # rows é lista (normalizada). Queremos um dict simples.
    data = rows[-1] if rows else {}
    # defaults mínimos para a UI
    data.setdefault("enabled", get_mode() == "sim")
    data.setdefault("rpm", 900)
    data.setdefault("speed", 0.0)
    data.setdefault("coolant", 88.0)
    data.setdefault("steer", 0.0)
    data.setdefault("throttle", 0.0)
    data.setdefault("brake", 0.0)
    data.setdefault("timestamp", int(time.time() * 1000))
    return jsonify(data)

@app.route("/sim_sensors/history")
def sim_history():
    try:
        n = int(request.args.get("n", "120"))
    except Exception:
        n = 120
    return jsonify(get_history(n))

@app.route("/sim_sensors/control", methods=["POST"])
def sim_control():
    payload = _safe_json()
    st = set_controls(
        throttle=payload.get("throttle"),
        brake=payload.get("brake"),
        steer=payload.get("steer"),
    )
    # `set_controls` pode não retornar estado; garanta um dict
    if not isinstance(st, dict):
        st = {"ok": True}
    return jsonify({"ok": True, "state": st})

# Alias compatível
@app.route("/sensors")
def sensors_latest():
    global _first_real_call_done
    if not _first_real_call_done:
        _first_real_call_done = True
        set_mode("real")
    rows = get_latest()
    return jsonify({
        "ok": True, 
        "readings": rows[0]["graph"], 
        "classes": rows[0]["classes"], 
        "running": is_running(), 
        "mode": get_mode()
    })

# -----------------------------------------------------------------------------
# RAG caches
# -----------------------------------------------------------------------------
_ASSETS_CACHE = {}
_INDEX_CACHE  = {}

def paths_for_model(slug: str):
    return os.path.join(EXTRACT_BASE, slug), os.path.join(INDEX_BASE, slug)

def ensure_model_loaded(slug: str):
    if slug not in _ASSETS_CACHE:
        extract_root, _ = paths_for_model(slug)
        if not os.path.isdir(extract_root):
            raise FileNotFoundError(f"Assets não encontrados para '{slug}': {extract_root}")
        _ASSETS_CACHE[slug] = load_assets_index(extract_root)
    if slug not in _INDEX_CACHE:
        _, index_root = paths_for_model(slug)
        bm25, docs_meta, _ = load_bm25_index(index_root)
        faiss_index, faiss_ids, _ = load_faiss_index(index_root)
        _INDEX_CACHE[slug] = (bm25, docs_meta, faiss_index, faiss_ids)

# -----------------------------------------------------------------------------
# LM Studio status
# -----------------------------------------------------------------------------
@app.route("/lm_status")
def lm_status():
    url = f"{LM_STUDIO_BASE}/v1/models"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=2.0) as resp:
            data = _json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
            loaded = bool(data.get("data"))
            return jsonify({"ok": True, "loaded": loaded})
    except (HTTPError, URLError, TimeoutError, Exception) as e:
        return jsonify({"ok": False, "loaded": False, "error": str(e)}), 503

ensure_idle_monitor()

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
@app.route("/")
def chat_ui():
    return render_template("chat.html")

# -----------------------------------------------------------------------------
# RAG endpoint
# -----------------------------------------------------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    vehicle_model = (data.get("vehicleModel") or "").strip()
    max_tokens = int(data.get("maxContextTokens") or 500)

    if not question:
        return jsonify({"answer": "Pergunta vazia.", "images": []})

    slug = normalize_model(vehicle_model)
    if not slug:
        return jsonify({"answer": "Selecione um modelo de veículo.", "images": []})

    try:
        ensure_model_loaded(slug)
    except FileNotFoundError as e:
        return jsonify({"answer": f"Modelo '{vehicle_model}' não disponível: {e}", "images": []})

    assets_index = _ASSETS_CACHE[slug]
    bm25, docs_meta, faiss_index, faiss_ids = _INDEX_CACHE[slug]

    answer, rel_images = run_rag_pipeline(
        user_query=question,
        vehicle_model=vehicle_model,
        assets_index=assets_index,
        bm25=bm25,
        docs_meta=docs_meta,
        faiss_index=faiss_index,
        faiss_ids=faiss_ids,
        max_context_tokens=max_tokens,
    )

    def _strip_prefixes(p: str, slug: str) -> str:
        p = (p or "").replace("\\", "/").lstrip("./")
        while p.startswith(slug + "/"):
            p = p[len(slug) + 1:]
        for prefix in ("assets_out/", "./assets_out/"):
            pref = f"{prefix}{slug}/"
            if p.startswith(pref):
                p = p[len(pref):]
        return p

    images = [f"/assets/{slug}/{_strip_prefixes(p, slug)}" for p in rel_images]
    return jsonify({"answer": answer, "images": images, "model": slug})

# -----------------------------------------------------------------------------
# Assets extraídos
# -----------------------------------------------------------------------------
@app.route("/assets/<model>/<path:filename>")
def serve_assets(model, filename):
    slug = normalize_model(model)
    extract_root, _ = paths_for_model(slug)
    if not os.path.isdir(extract_root):
        abort(404)

    fn = (filename or "").replace("\\", "/").lstrip("./")
    fn = fn.replace("..", "")
    while fn.startswith(slug + "/"):
        fn = fn[len(slug) + 1:]
    for prefix in ("assets_out/", "./assets_out/"):
        pref = f"{prefix}{slug}/"
        if fn.startswith(pref):
            fn = fn[len(pref):]

    return send_from_directory(extract_root, fn, as_attachment=False)

# -----------------------------------------------------------------------------

def main():
    port = int(os.environ.get("PORT", "5001"))
    debug = bool(int(os.environ.get("DEBUG", "0")))
    print(f"* EXTRACT_BASE: {EXTRACT_BASE}")
    print(f"* INDEX_BASE  : {INDEX_BASE}")
    print(f"* Telemetry   : mode={get_mode()} running={is_running()}")

    start_background(mode=CAN_SOURCE, hz=CAN_HZ)
    app.run(host="0.0.0.0", port=port, debug=debug)

if __name__ == "__main__":
    main()
