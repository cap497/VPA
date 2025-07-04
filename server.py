from flask import Flask, request, jsonify
import threading
import time
import sys

from query_core import load_bm25_index, load_embedder, run_rag_pipeline, \
                        ensure_model_loaded, unload_model

# =========================
# CONFIG
# =========================
DEFAULT_IDLE_TIMEOUT_MINUTES = 30

app = Flask(__name__)
last_access_time = time.time()
idle_timeout_seconds = DEFAULT_IDLE_TIMEOUT_MINUTES * 60
should_shutdown = False

# =========================
# Load at startup
# =========================
print("🔄 Starting RAG server. Loading models and index...")
ensure_model_loaded()
bm25, sections, texts = load_bm25_index()
embedder = load_embedder()
print("✅ RAG server ready.")


# =========================
# ROUTES
# =========================
@app.route("/ask", methods=["POST"])
def ask():
    global last_access_time
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question'"}), 400

    question = data["question"]
    last_access_time = time.time()

    try:
        answer = run_rag_pipeline(
            bm25, sections, embedder, question
        )
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        return jsonify({"error": "Internal error"}), 500

@app.route("/set_timeout", methods=["POST"])
def set_timeout():
    global idle_timeout_seconds
    data = request.get_json()
    if not data or "minutes" not in data:
        return jsonify({"error": "Missing 'minutes'"}), 400

    try:
        minutes = int(data["minutes"])
        idle_timeout_seconds = minutes * 60
        return jsonify({"status": f"Idle timeout set to {minutes} minutes."})
    except Exception:
        return jsonify({"error": "Invalid 'minutes' value"}), 400


# =========================
# Background idle monitor
# =========================
def idle_monitor():
    global should_shutdown
    while not should_shutdown:
        time.sleep(5)
        idle_duration = time.time() - last_access_time
        if idle_duration > idle_timeout_seconds:
            print(f"⏰ Idle for {idle_duration:.1f}s (limit {idle_timeout_seconds}s). Shutting down.")
            should_shutdown = True
            unload_model()
            agent_loaded = False
            sys.exit(0)


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    threading.Thread(target=idle_monitor, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
