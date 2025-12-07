# sensors/can_api.py
from flask import Blueprint, jsonify, request
from .can_store import CANStore

can_blueprint = Blueprint("can_api", __name__)

# store é injetado por app.py via atributo
def _store() -> CANStore:
    return can_blueprint.store  # type: ignore[attr-defined]

@can_blueprint.route("/can/health")
def health():
    return jsonify({"ok": True})

@can_blueprint.route("/can/stats")
def stats():
    return jsonify(_store().stats())

@can_blueprint.route("/can/readings")
def readings():
    limit = int(request.args.get("limit", "50"))
    return jsonify(_store().get_readings(limit=limit))
