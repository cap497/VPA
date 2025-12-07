#!/usr/bin/env bash

cleanup() {
    echo -e "\n🛑 Encerrando todos os processos da simulação..."
    pkill -f "server.py" || true
    pkill -f "client.py" || true
    echo "✅ Processos encerrados."
}
trap cleanup INT TERM

if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

# Parâmetros FIXOS do experimento (não sobrescritos por ambiente)
CLIENT_ALPHA=0.01
CLIENT_EPOCHS=25
SERVER_MIX=0.6
ROUNDS=25

echo "⚙️  Hiperparâmetros (fixos): ROUNDS=$ROUNDS, SERVER_MIX=$SERVER_MIX, CLIENT_ALPHA=$CLIENT_ALPHA, CLIENT_EPOCHS=$CLIENT_EPOCHS"

# 1) Preparar dados a partir do analyzed_data_drivers.csv (se existir)
ANALYZED="./drivers/analyzed_data_drivers.csv"
if [ -f "$ANALYZED" ]; then
    echo "📦 Preparando dados a partir de $ANALYZED ..."
    $PY prepare_data.py --source "$ANALYZED" || { echo "❌ Falha ao preparar dados a partir do analyzed_data_drivers.csv"; exit 1; }
else
    echo "⚠️  $ANALYZED não encontrado; usando divisão sintética (10 clientes)."
    $PY prepare_data.py || { echo "❌ Falha ao preparar dados (divisão sintética)"; exit 1; }
fi

echo "🔁 Rodando avaliação centralizada (gera relatorio_centralizado.csv)..."
$PY centralized.py --init random --epochs $CLIENT_EPOCHS
echo "→ Centralizado concluído."

run_strategy() {
    local STRAT=$1
    local MU_ARG=$2
    echo "🚀 Iniciando o servidor federado ($STRAT)..."
    $PY server.py --init random --server-mix $SERVER_MIX --rounds $ROUNDS --strategy "$STRAT" $MU_ARG &
    SERVER_PID=$!
    echo "   Servidor ($STRAT) iniciado com PID: $SERVER_PID"
    echo "   Aguardando o servidor ficar pronto..."
    sleep 3
    CLIENT_IDS=({A..J})
    echo "🚀 Iniciando ${#CLIENT_IDS[@]} clientes..."
    for id in "${CLIENT_IDS[@]}"; do
            $PY client.py --id "$id" --alpha $CLIENT_ALPHA --epochs $CLIENT_EPOCHS --shuffle &
            echo "   Cliente $id iniciado."
            sleep 0.2
    done
    echo -e "\n🎉 Simulação federada ($STRAT) em andamento com 1 servidor e 10 clientes."
    wait $SERVER_PID
}

# Rodar FedAvg
run_strategy "FedAvg"

# Rodar FedProx (mu padrão da estratégia; para sobrescrever: export MU=0.1)
if [ -n "$MU" ]; then MU_ARG="--mu $MU"; else MU_ARG=""; fi
run_strategy "FedProx" "$MU_ARG"

echo "🔁 Simulações federadas concluídas. Gerando comparação central vs federado..."
$PY evaluate_comparison.py
echo "→ Comparação salva em ./graficos (e arquivos em ./resultados)."

