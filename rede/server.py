import socket
import struct
import threading
import numpy as np
import pandas as pd
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
import os
import argparse

# --- Configurações ---
HOST, PORT = '127.0.0.1', 9000
NUM_CLIENTS, NUM_ROUNDS, NUM_RULES, NUM_FEATURES = 10, 25, 10, 5
NUM_CLUSTERS = 3

# --- Protocolo ---
HEADER_FORMAT_RECV = '<HBII'
HEADER_FORMAT_SEND = '<HBII'
MAGIC_HEADER, OPCODE_SEND, OPCODE_RECV = 0xABCD, 1, 2

client_updates, client_data_sizes = {}, {}
clients_ready, lock = threading.Event(), threading.Lock()
GLOBAL_SCALER = None
# --- Normalization utilities (inlined from fixed_normalization.py) ---
FEATURES = [
    'speed',
    'acc_norm',
    'engine_speed',
    'throttle_position',
    'delta_acc_lat',
]
MIN_VALUES = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
MAX_VALUES = np.array([120.0, 5.0, 10000.0, 100.0, 3.0], dtype=np.float32)
SCALE_VALUES = 1.0 / (MAX_VALUES - MIN_VALUES)

def normalize_np(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.clip((X - MIN_VALUES) * SCALE_VALUES, 0.0, 1.0)

def normalize_df(df: pd.DataFrame) -> np.ndarray:
    return normalize_np(df[FEATURES].values)

def get_scaler_like():
    class _Scaler: pass
    s = _Scaler()
    s.data_min_ = MIN_VALUES.astype(np.float32)
    s.scale_ = SCALE_VALUES.astype(np.float32)
    return s

# --- Parameter loader (inlined from params_loader.py) ---
import io, zipfile
from typing import Optional, Dict

def _load_from_dir(dir_path: str, num_features: int, num_rules: int) -> Optional[Dict[str, np.ndarray]]:
    c_path = os.path.join(dir_path, 'c.csv')
    s_path = os.path.join(dir_path, 's.csv')
    p_path = os.path.join(dir_path, 'p.csv')
    q_path = os.path.join(dir_path, 'q.csv')
    if all(os.path.exists(p) for p in [c_path, s_path, p_path, q_path]):
        try:
            c = pd.read_csv(c_path, header=None).values
            s = pd.read_csv(s_path, header=None).values
            p = pd.read_csv(p_path, header=None).values
            q = pd.read_csv(q_path, header=None).values.squeeze()
            if c.size != num_features * num_rules or s.size != num_features * num_rules or p.size != num_features * num_rules or q.size != num_rules:
                return None
            c = c.reshape(num_features, num_rules)
            s = s.reshape(num_features, num_rules)
            p = p.reshape(num_features, num_rules)
            q = q.reshape(num_rules)
            return {'c': c.astype(float), 's': s.astype(float), 'p': p.astype(float), 'q': q.astype(float)}
        except Exception:
            return None
    return None

def _load_from_npz(npz_path: str, num_features: int, num_rules: int) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(npz_path) as data:
            c = data.get('c'); s = data.get('s'); p = data.get('p'); q = data.get('q')
            if c is None or s is None or p is None or q is None:
                return None
            c = np.asarray(c); s = np.asarray(s); p = np.asarray(p); q = np.asarray(q)
            if c.size != num_features * num_rules or s.size != num_features * num_rules or p.size != num_features * num_rules or q.size != num_rules:
                return None
            c = c.reshape(num_features, num_rules)
            s = s.reshape(num_features, num_rules)
            p = p.reshape(num_features, num_rules)
            q = q.reshape(num_rules)
            return {'c': c, 's': s, 'p': p, 'q': q}
    except Exception:
        return None

def _load_from_zip(zip_path: str, num_features: int, num_rules: int) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(zip_path):
        return None
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = {name.lower(): name for name in zf.namelist()}
            required = ['c.csv', 's.csv', 'p.csv', 'q.csv']
            if not all(r in members for r in required):
                return None
            def read_csv(name):
                with zf.open(members[name]) as f:
                    return pd.read_csv(io.TextIOWrapper(f, encoding='utf-8'), header=None).values
            c = read_csv('c.csv'); s = read_csv('s.csv'); p = read_csv('p.csv'); q = read_csv('q.csv')
            if c.size != num_features * num_rules or s.size != num_features * num_rules or p.size != num_features * num_rules or q.size != num_rules:
                return None
            c = c.reshape(num_features, num_rules)
            s = s.reshape(num_features, num_rules)
            p = p.reshape(num_features, num_rules)
            q = q.reshape(num_rules)
            return {'c': c.astype(float), 's': s.astype(float), 'p': p.astype(float), 'q': q.astype(float)}
    except Exception:
        return None

def load_initial_params(num_features: int, num_rules: int, prefer_paths: Optional[list] = None) -> Optional[Dict[str, np.ndarray]]:
    paths = prefer_paths[:] if prefer_paths else []
    paths += [
        os.path.join('rede','parametros_globais'),
        os.path.join('rede','resultados', 'FedAvg', 'final_params_FedAvg.npz'),
        os.path.join(os.path.dirname(__file__), '..', 'rede.zip'),
        os.path.join(os.path.dirname(__file__), 'rede.zip'),
    ]
    for p in paths:
        if os.path.isdir(p):
            loaded = _load_from_dir(p, num_features, num_rules)
            if loaded: return loaded
        elif os.path.isfile(p):
            if p.lower().endswith('.npz'):
                loaded = _load_from_npz(p, num_features, num_rules)
                if loaded: return loaded
            elif p.lower().endswith('.zip'):
                loaded = _load_from_zip(p, num_features, num_rules)
                if loaded: return loaded
    return None

def calys(x, num_inputs, num_rules, centers, sigmas, weights, biases):
    rule_outputs = biases + np.dot(x, weights)
    diff = x[:, None] - centers
    exponent = -0.5 * (diff ** 2) / (sigmas ** 2)
    rule_weights = np.exp(exponent).prod(axis=0)
    numerator = np.sum(rule_weights * rule_outputs)
    denominator = np.sum(rule_weights)
    return numerator / (denominator + 1e-8)

def initialize_global_model(init_mode='random'):
    params = None
    if init_mode == 'load':
        params = load_initial_params(NUM_FEATURES, NUM_RULES)
    if params is None:
        params = {
            'c': np.random.rand(NUM_FEATURES, NUM_RULES),
            's': 0.2 + 0.3 * np.random.rand(NUM_FEATURES, NUM_RULES),
            'p': np.random.randn(NUM_FEATURES, NUM_RULES) * 0.1,
            'q': np.random.randn(NUM_RULES) * 0.1,
        }
    model = {**params, 'centroids': np.random.rand(NUM_CLUSTERS, NUM_FEATURES)}
    return model

def pack_weights(params, include_centroids=True):
    parts = [params['c'].flatten(), params['s'].flatten(), params['p'].flatten(), params['q'].flatten()]
    if include_centroids and 'centroids' in params and params['centroids'] is not None:
        parts.append(params['centroids'].flatten())
    return np.concatenate(parts).astype(np.float32)

def unpack_weights(weights_flat):
    shapes = ((NUM_FEATURES, NUM_RULES), (NUM_FEATURES, NUM_RULES), (NUM_FEATURES, NUM_RULES), (NUM_RULES,))
    c_s, s_s, p_s = np.prod(shapes[0]), np.prod(shapes[1]), np.prod(shapes[2])
    base_len = c_s + s_s + p_s + shapes[3][0]
    centroid_len = NUM_CLUSTERS * NUM_FEATURES

    res = {}
    res['c'] = weights_flat[:c_s].reshape(shapes[0])
    res['s'] = weights_flat[c_s:c_s + s_s].reshape(shapes[1])
    res['p'] = weights_flat[c_s + s_s:c_s + s_s + p_s].reshape(shapes[2])
    res['q'] = weights_flat[c_s + s_s + p_s:base_len]

    if weights_flat.size >= base_len + centroid_len:
        res['centroids'] = weights_flat[base_len:base_len + centroid_len].reshape((NUM_CLUSTERS, NUM_FEATURES))
    else:
        res['centroids'] = None

    return res

def fed_avg_aggregator(client_updates_local, client_data_sizes_local):
    total_data_size = sum(client_data_sizes_local.values())
    if not client_updates_local or total_data_size == 0:
        return None
    first_params = client_updates_local[next(iter(client_updates_local))]['params']
    agg_params = {}
    for k, v in first_params.items():
        if v is None:
            continue
        try:
            agg_params[k] = np.zeros_like(v, dtype=float)
        except Exception:
            continue

    for cid, update in client_updates_local.items():
        weight = client_data_sizes_local.get(cid, 0) / total_data_size
        for k in list(agg_params.keys()):
            val = update['params'].get(k)
            if val is None:
                continue
            agg_params[k] += np.array(val, dtype=float) * weight

    return agg_params

STRATEGIES = {
    'FedAvg': {'aggregator': fed_avg_aggregator, 'mu': 0.0},
    'FedProx': {'aggregator': fed_avg_aggregator, 'mu': 0.1},
}

def _stabilize_params(params):
    try:
        params['c'] = np.clip(params['c'], 0.0, 1.0)
        params['s'] = np.clip(params['s'], 1e-3, 10.0)
        params['p'] = np.clip(params['p'], -5.0, 5.0)
        params['q'] = np.clip(params['q'], -5.0, 5.0)
    except Exception:
        pass
    return params

def _all_finite(params):
    try:
        return all(np.all(np.isfinite(v)) for v in [params.get('c'), params.get('s'), params.get('p'), params.get('q')])
    except Exception:
        return False
def evaluate_round(agg_params, client_updates_local, X_test, y_test, round_num, strategy_name):
    y_pred_val = np.array([
        calys(
            X_test[i],
            NUM_FEATURES,
            NUM_RULES,
            agg_params['c'],
            agg_params['s'],
            agg_params['p'],
            agg_params['q']
        ) for i in range(len(X_test))
    ])
    y_pred_val_rounded = np.clip(np.round(y_pred_val), 1, 3)
    val_metrics = {
        'val_accuracy': accuracy_score(y_test, y_pred_val_rounded),
        'val_mse': mean_squared_error(y_test, y_pred_val),
        'val_f1_score': f1_score(y_test, y_pred_val_rounded, average='weighted', zero_division=0),
        'val_precision': precision_score(y_test, y_pred_val_rounded, average='weighted', zero_division=0),
        'val_recall': recall_score(y_test, y_pred_val_rounded, average='weighted', zero_division=0)
    }
    total_data_size = sum(client_data_sizes.values())
    avg_train_metrics = {m: 0 for m in ['mse', 'accuracy', 'f1_score']}
    for cid, update in client_updates_local.items():
        weight = client_data_sizes[cid] / total_data_size
        for m in avg_train_metrics:
            avg_train_metrics[m] += update['metrics'][m] * weight

    report = {'round': round_num, 'strategy': strategy_name, **{f'train_{k}':v for k,v in avg_train_metrics.items()}, **val_metrics}
    conf_matrix = confusion_matrix(y_test, y_pred_val_rounded, labels=[1, 2, 3])

    print(f"\n--- AVALIAÇÃO - {strategy_name.upper()} - RODADA {round_num} ---")
    print(f"Train Médio: Acc: {report['train_accuracy']:.4f}, MSE: {report['train_mse']:.4f}")
    print(f"Validação Global: Acc: {report['val_accuracy']:.4f}, MSE: {report['val_mse']:.4f}")
    return report, conf_matrix

def run_federated_learning(strategy_name, connections, X_test, y_test, init_mode='random', server_mix: float = 1.0, mu_override: float | None = None, rounds: int | None = None):
    strategy_config = STRATEGIES[strategy_name]
    aggregator_func = strategy_config['aggregator']
    mu = strategy_config['mu'] if mu_override is None else float(mu_override)

    print(f"\n\n--- INICIANDO {strategy_name.upper()} (mu={mu}) ---")
    global_params = initialize_global_model(init_mode=init_mode)
    history = []
    client_metrics = {}
    centroid_history = []
    base_out = os.path.join('resultados', strategy_name)
    anfis_out = os.path.join(base_out, 'anfis')
    cent_out = os.path.join(base_out, 'centroids')
    graficos_out = 'graficos'
    os.makedirs(anfis_out, exist_ok=True)
    os.makedirs(cent_out, exist_ok=True)
    os.makedirs(graficos_out, exist_ok=True)
    with lock:
        dummy_client_updates = {}
        report, conf_matrix = evaluate_round(global_params, dummy_client_updates, X_test, y_test, 0, strategy_name)
        if report:
            history.append(report)
        data = pd.read_csv('drivers/clustered_data_federated.csv')
        features = FEATURES
        for cid in client_data_sizes:
            client_data = data[data['client'] == cid]
            X_client = normalize_df(client_data[features])
            y_client = client_data['cluster_id'].values
            y_pred = np.array([
                calys(
                    X_client[i],
                    NUM_FEATURES,
                    NUM_RULES,
                    global_params['c'],
                    global_params['s'],
                    global_params['p'],
                    global_params['q']
                ) for i in range(len(X_client))
            ])
            mse = mean_squared_error(y_client, y_pred)
            y_pred_rounded = np.clip(np.round(y_pred), 1, 3)
            accuracy = accuracy_score(y_client, y_pred_rounded)
            f1 = f1_score(y_client, y_pred_rounded, average='weighted', zero_division=0)
            client_metrics[cid] = {'round': [0], 'mse': [mse], 'accuracy': [accuracy], 'f1_score': [f1]}

    total_rounds = NUM_ROUNDS if rounds is None else int(rounds)
    for round_num in range(1, total_rounds + 1):
        print(f"\n--- Rodada {round_num}/{total_rounds} ---")
        with lock:
            client_updates.clear()
            clients_ready.clear()

            core_weights = pack_weights(global_params, include_centroids=True).tobytes()
            scaler_like = get_scaler_like()
            scaler_min = scaler_like.data_min_.astype(np.float32)
            scaler_scale = scaler_like.scale_.astype(np.float32)
            scaler_bytes = scaler_min.tobytes() + scaler_scale.tobytes()
            weights_payload = struct.pack('<B', 1) + scaler_bytes + core_weights

            header = struct.pack(HEADER_FORMAT_SEND, MAGIC_HEADER, OPCODE_SEND, len(weights_payload), 0)
            mu_bytes = struct.pack('<f', mu)

            for conn in connections:
                try:
                    conn.sendall(header + mu_bytes + weights_payload)
                except Exception:
                    pass
        print("Modelo global enviado.")
        if not clients_ready.wait(timeout=60.0):
            print("Timeout!")
            break

        with lock:
            print(f"Atualizações recebidas: {len(client_updates)}/{len(client_data_sizes)} clientes")
            aggregated_params = aggregator_func(client_updates, client_data_sizes)
            if aggregated_params:
                aggregated_params = _stabilize_params(aggregated_params)
                if not _all_finite(aggregated_params):
                    print("Aviso: parâmetros agregados contêm NaN/Inf. Mantendo modelo global anterior nesta rodada.")
                else:
                    base_mix = float(max(0.0, min(1.0, server_mix)))
                    mix = 1.0 if round_num == 1 else base_mix
                    for k, v in aggregated_params.items():
                        if k in ('c', 's', 'p', 'q') and v is not None and global_params.get(k) is not None:
                            global_params[k] = (1.0 - mix) * global_params[k] + mix * v
                        else:
                            global_params[k] = v

            total_counts = np.zeros(NUM_CLUSTERS)
            sum_vectors = np.zeros((NUM_CLUSTERS, NUM_FEATURES))
            for cid, update in client_updates.items():
                cent_info = update['metrics'].get('centroid_info')
                if cent_info:
                    counts = np.array(cent_info.get('counts', [0]*NUM_CLUSTERS), dtype=float)
                    sums = np.array(cent_info.get('sums', [[0.0]*NUM_FEATURES]*NUM_CLUSTERS), dtype=float)
                    total_counts += counts
                    sum_vectors += sums

            if total_counts.sum() > 0:
                new_centroids = np.zeros_like(global_params['centroids'])
                for k in range(NUM_CLUSTERS):
                    if total_counts[k] > 0:
                        new_centroids[k] = sum_vectors[k] / total_counts[k]
                    else:
                        new_centroids[k] = global_params['centroids'][k]
                global_params['centroids'] = new_centroids

            centroid_history.append({
                'round': round_num,
                'total_counts': total_counts.tolist(),
                'centroids': global_params['centroids'].tolist(),
                'sum_squares': float(np.nansum([u['metrics'].get('centroid_info', {}).get('sse', 0.0) for u in client_updates.values()])),
            })

            report, conf_matrix = evaluate_round(global_params, client_updates, X_test, y_test, round_num, strategy_name)
            for cid in client_metrics:
                client_metrics[cid]['round'].append(round_num)
                if cid in client_updates:
                    client_metrics[cid]['mse'].append(client_updates[cid]['metrics'].get('mse', np.nan))
                    client_metrics[cid]['accuracy'].append(client_updates[cid]['metrics'].get('accuracy', np.nan))
                    client_metrics[cid]['f1_score'].append(client_updates[cid]['metrics'].get('f1_score', np.nan))
                else:
                    client_metrics[cid]['mse'].append(np.nan)
                    client_metrics[cid]['accuracy'].append(np.nan)
                    client_metrics[cid]['f1_score'].append(np.nan)
        if report: history.append(report)

    if not history:
        print("Nenhuma rodada completada."); return

    history_df = pd.DataFrame(history)
    history_csv = os.path.join(anfis_out, f'relatorio_{strategy_name}.csv')
    history_df.to_csv(history_csv, index=False)
    print(f"Relatório salvo em '{history_csv}'")
    
    cent_df = pd.DataFrame([{
        'round': h['round'],
        'total_counts': h['total_counts'],
        'sum_squares': h['sum_squares'],
        'centroids': h['centroids']
    } for h in centroid_history])
    cent_csv = os.path.join(cent_out, 'centroid_history.csv')
    cent_df.to_csv(cent_csv, index=False)
    print(f"Histórico de centroides salvo em '{cent_csv}'")

    final_params_path = os.path.join(base_out, f'final_params_{strategy_name}.npz')
    try:
        np.savez_compressed(final_params_path, centroids=global_params.get('centroids'), c=global_params.get('c'), s=global_params.get('s'), p=global_params.get('p'), q=global_params.get('q'))
        print(f"Parâmetros finais salvos em '{final_params_path}'")
    except Exception as e:
        print('Falha ao salvar parâmetros finais:', e)

    for cid, metrics in client_metrics.items():
        if not metrics['round']:
            continue
        plt.figure(figsize=(12, 6))
        plt.plot(metrics['round'], metrics['mse'], 'bo-', label='MSE')
        plt.plot(metrics['round'], metrics['accuracy'], 'go-', label='Acurácia')
        plt.plot(metrics['round'], metrics['f1_score'], 'ro-', label='F1-Score')
        plt.title(f'Evolução das Métricas - Cliente {cid} - {strategy_name}')
        plt.xlabel('Rodada')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True, linestyle='--')
        try:
            max_round = int(np.nanmax(metrics['round'])) if metrics['round'] else total_rounds
        except Exception:
            max_round = total_rounds
        plt.xticks(range(0, max_round + 1))
        plt.tight_layout()
        out_path = os.path.join(graficos_out, f'grafico_cliente_{cid}_{strategy_name}.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Gráfico do cliente {cid} salvo em '{out_path}'")

    # --- Consolidated chart: F1-score per client (A–J) ---
    try:
        desired_clients = [chr(ord('A') + i) for i in range(10)]
        f1_final = {}
        for cid in desired_clients:
            m = client_metrics.get(cid)
            if not m:
                f1_final[cid] = np.nan
                continue
            arr = np.array(m.get('f1_score', []), dtype=float)
            # pega o último valor não-NaN
            if arr.size == 0 or np.all(np.isnan(arr)):
                f1_final[cid] = np.nan
            else:
                # índice do último não-NaN
                idxs = np.where(~np.isnan(arr))[0]
                f1_final[cid] = float(arr[idxs[-1]]) if idxs.size > 0 else np.nan

        labels = list(f1_final.keys())
        values = [f1_final[k] for k in labels]
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color='#E45756')
        plt.ylim(0, 1)
        plt.xlabel('Client')
        plt.ylabel('F1-score (weighted)')
        plt.title(f'F1-score per client (A–J) - {strategy_name}')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        for b, v in zip(bars, values):
            txt = 'N/A' if np.isnan(v) else f"{v:.3f}"
            plt.text(b.get_x() + b.get_width()/2, (0 if np.isnan(v) else v) + 0.02, txt,
                     ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(graficos_out, f'f1_clients_A_J_{strategy_name}.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Consolidated F1 (A–J) chart saved at '{out_path}'")
    except Exception as e:
        print(f"[Warning] Failed to generate consolidated F1 A–J chart: {e}")

def handle_client(conn, addr):
    client_id = None
    try:
        id_len = struct.unpack('<I', conn.recv(4))[0]; client_id = conn.recv(id_len).decode()
        data_size = struct.unpack('<I', conn.recv(4))[0]
        with lock: client_data_sizes[client_id] = data_size
        print(f"Cliente {client_id} conectado com {data_size} amostras.")

        while True:
            header_bytes = conn.recv(struct.calcsize(HEADER_FORMAT_RECV))
            if not header_bytes: break
            
            magic, opcode, w_len, m_len = struct.unpack(HEADER_FORMAT_RECV, header_bytes)
            if magic != MAGIC_HEADER or opcode != OPCODE_RECV: continue
            
            weights_bytes = conn.recv(w_len)
            metrics_bytes = conn.recv(m_len)
            
            weights_flat = np.frombuffer(weights_bytes, dtype=np.float32)
            params = unpack_weights(weights_flat)
            
            with lock:
                metrics = json.loads(metrics_bytes.decode()) if metrics_bytes else {}
                client_updates[client_id] = {'params': params, 'metrics': metrics}
                if len(client_updates) >= len(client_data_sizes): clients_ready.set()
    except Exception: pass
    finally:
        with lock:
            if client_id in client_data_sizes: del client_data_sizes[client_id]
            if client_id in client_updates: del client_updates[client_id]
        conn.close(); print(f"Conexão com {client_id} encerrada.")

def main(init_mode='random', server_mix: float = 1.0, rounds: int | None = None, strategy: str = 'FedAvg', mu: float | None = None):
    data = pd.read_csv('drivers/clustered_data_federated.csv')
    features = FEATURES
    global GLOBAL_SCALER
    GLOBAL_SCALER = get_scaler_like()
    X = normalize_df(data[features])
    y = data['cluster_id'].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT)); server_socket.listen(NUM_CLIENTS)
    print(f"Servidor escutando em {HOST}:{PORT}")
    
    connections = []
    threads = []
    try:
        while len(connections) < NUM_CLIENTS:
            conn, addr = server_socket.accept()
            connections.append(conn)
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()
            threads.append(thread)
        
        while True:
            with lock:
                if len(client_data_sizes) == NUM_CLIENTS: 
                    print(f"\nTodos os {NUM_CLIENTS} clientes conectados. Iniciando o processo federado.")
                    break
            time.sleep(0.5)
            
        strat = strategy if strategy in STRATEGIES else 'FedAvg'
        run_federated_learning(
            strat,
            connections,
            X_test,
            y_test,
            init_mode=init_mode,
            server_mix=server_mix,
            mu_override=mu,
            rounds=rounds,
        )
    
    finally:
        print("\nEncerrando o servidor e as conexões...")
        for conn in connections:
            conn.close()
        server_socket.close()
        for t in threads:
            t.join(timeout=1.0)
        print("\nProcesso de aprendizado federado concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', choices=['random', 'load'], default='random', help='How to initialize global model parameters')
    parser.add_argument('--server-mix', type=float, default=0.3, help='Server aggregation mixing factor (0..1): 1 replaces, <1 smooths updates')
    parser.add_argument('--rounds', type=int, default=25, help='Number of federated rounds to run')
    parser.add_argument('--strategy', choices=['FedAvg', 'FedProx'], default='FedAvg', help='Federated strategy to run')
    parser.add_argument('--mu', type=float, default=None, help='Override proximal term mu (None uses strategy default)')
    args = parser.parse_args()
    main(init_mode=args.init,
        server_mix=args.server_mix if hasattr(args, 'server_mix') else 0.3,
        rounds=args.rounds if hasattr(args, 'rounds') else None,
        strategy=args.strategy if hasattr(args, 'strategy') else 'FedAvg',
        mu=args.mu if hasattr(args, 'mu') else None)
