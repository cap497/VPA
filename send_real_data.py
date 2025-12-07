import json
import paho.mqtt.client as mqtt
import time
import csv
import os

client = mqtt.Client()
client.connect("localhost", 1883, 60)

# Caminho para o CSV
csv_path = os.path.join('rede', 'drivers', 'analyzed_data_drivers.csv')

with open(csv_path, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # Monta payload convertendo valores do CSV para float/int
        payload = {
            "speed": float(row.get("speed", 0)),
            "acc_long": float(row.get("acc_long", 0)),
            "acc_lat": float(row.get("acc_lat", 0)),
            "acc_norm": float(row.get("acc_norm", 0)),
            "engine_speed": int(float(row.get("engine_speed", 0))),
            "throttle_position": float(row.get("throttle_position", 0)),
            "jerk": float(row.get("delta_acc_lat", 0))
        }

        msg = json.dumps(payload, ensure_ascii=False)
        client.publish("futurelab/can", msg)
        print("Enviado:", payload)

        time.sleep(0.5)  # intervalo entre cada linha