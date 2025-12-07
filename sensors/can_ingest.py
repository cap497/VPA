# sensors/can_ingest.py
import os, json, time, threading
from typing import Optional
from .can_store import CANStore

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None  # opcional; só necessário se usar MQTT de verdade

MQTT_URL = os.environ.get("CAN_MQTT_URL", "mqtt://localhost:1883")
MQTT_HOST = os.environ.get("CAN_MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("CAN_MQTT_PORT", "1883"))
MQTT_TOPIC = os.environ.get("CAN_MQTT_TOPIC", "futurelab/can")

class CANIngestWorker:
    def __init__(self, store: CANStore, enabled: bool = True):
        self.store = store
        self.enabled = enabled and (mqtt is not None)
        self._th: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if not self.enabled:
            print("[CAN-INGEST] MQTT disabled (paho-mqtt not installed or flag off).")
            return
        if self._th and self._th.is_alive(): return
        self._th = threading.Thread(target=self._run, daemon=True)
        print(f"[CAN-INGEST] Subscribing MQTT {MQTT_HOST}:{MQTT_PORT} topic={MQTT_TOPIC}...")
        self._th.start()
        

    def stop(self):
        self._stop.set()

    def _run(self):
        client = mqtt.Client()
        def on_connect(c, u, f, rc): 
            result, _ = c.subscribe(MQTT_TOPIC)
            if result == 0:
                print(f"[CAN-INGEST] Subscribed MQTT {MQTT_HOST}:{MQTT_PORT} topic={MQTT_TOPIC}")
            else:
                print(f"[CAN-INGEST][ERRO] Falha ao subscrever tópico {MQTT_TOPIC} (código={result})")

        def on_message(c, u, msg):
            ts = int(time.time() * 1000)
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
            except Exception:
                payload = {"raw": msg.payload.decode("utf-8", "ignore")}
            print(f"[CAN-INGEST] Message: {payload}")

            can_id = str(payload.get("id") or payload.get("can_id") or payload.get("message_id") or "unknown")
            self.store.add_reading(ts, can_id, payload)
        
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(MQTT_HOST, MQTT_PORT, 60)
        client.loop_start()
        while not self._stop.is_set():
            time.sleep(1)
        client.loop_stop()
        client.disconnect()
