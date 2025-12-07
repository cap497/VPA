# Vehicle Manual RAG — Pro

Mantém **BM25 → FAISS → Reranker (embeddings)**, com **orçamento de tokens**, integração opcional com **LM Studio**, e **sensores** (CAN-API + simulados).

---

## Índices offline (pré-processamento dos manuais)

Este app **não processa PDFs diretamente**.
Ele usa pastas extraídas em `assets_out/<modelo>` contendo `manifest.jsonl`, `text/`, `images/`, `table_images/`.

Para **cada carro**, é necessário rodar o `start.py` e gerar índices no diretório `indices/<modelo>`:

```bash
# Exemplo para Fiat Argo
python start.py argo

# Outros modelos suportados
python start.py hb20
python start.py kwid
python start.py onix
python start.py tcross
```

> ⚠️ Certifique-se de que os assets extraídos de cada manual estão em `assets_out/<modelo>/`.

---

## Rodar o app

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\activate

pip install -r requirements.txt

# Configuração de variáveis
export EXTRACT_BASE="./assets_out"
export INDEX_BASE="./indices"

# Opcional: LM Studio local (OpenAI API compatível)
# export OPENAI_API_BASE="http://localhost:1234/v1"
# export TARGET_MODEL_NAME="meta-llama-3-8b-instruct"

# Sensores reais (CAN-API)
# export CAN_SOURCE="real"
# export MQTT_URL="mqtt://localhost:1883"
# export MQTT_TOPIC="futurelab/can"
# export CAN_HZ=10

# Inicie o servidor Flask
python app.py
```

Abra no navegador: [http://localhost:5001](http://localhost:5001)

---

## Como funciona

1. **BM25** (recall) → top 200 docs.
2. **FAISS** (ANN) → top 200 docs → intersecção com BM25.
3. **Reranker** (SentenceTransformer) → top 20 docs.
4. **Seleção** de trechos respeitando orçamento de tokens (default: 500).
5. **Geração**: LM Studio (se disponível); fallback: resposta extrativa.
6. **Imagens** das páginas relevantes são retornadas com a resposta.

---

## UI

* Botões para escolher o **modelo do veículo** (Argo, HB20, Kwid, Onix, T-Cross).
* Campo para ajustar **orçamento de tokens**.
* Painel de **Sensores**:

  * **CAN-API** real: botão “Atualizar CAN-API”.
  * **Simulados**: botão “Alternar Simulados” e leitura periódica.

---

## Estrutura esperada

```
assets_out/
  argo/
  hb20/
  kwid/
  onix/
  tcross/
indices/
  argo/
  hb20/
  kwid/
  onix/
  tcross/
```

---

## Adicionando um novo carro

1. Extraia o manual para `assets_out/<novo_modelo>/`.
2. Gere os índices:

   ```bash
   python start.py <novo_modelo>
   ```
3. Reinicie o app.

Pronto! O novo modelo ficará disponível na interface.