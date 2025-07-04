from flask import Flask, Response, request
from twilio.twiml.messaging_response import MessagingResponse
import requests
import html

app = Flask(__name__)

RAG_SERVER_URL = "http://127.0.0.1:5000/ask"

@app.route("/twilio_webhook", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.form.get("Body")
    sender = request.form.get("From")
    print(f"📩 Mensagem recebida de {sender}: {incoming_msg}")

    try:
        rag_response = requests.post(
            RAG_SERVER_URL,
            json={"question": incoming_msg},
            timeout=60
        )
        rag_response.raise_for_status()
        answer = rag_response.json().get("answer", "Desculpe, não entendi sua pergunta.")
    except Exception as e:
        print(f"❌ Erro ao chamar o RAG: {e}")
        answer = "Desculpe, tive um problema para processar sua pergunta."

    #answer = html.escape(answer)
    print(f"➡️ Resposta enviada:\n{answer}")
    response = MessagingResponse()
    response.message(answer)
    final_response = str(response)
    return Response(
        final_response,
        content_type="application/xml"
    )

if __name__ == "__main__":
    app.run(port=5001, debug=True)
