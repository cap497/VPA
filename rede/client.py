import socket
import struct
import pandas as pd
import numpy as np
import argparse
import json
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
# --- Normalization utilities (inlined) ---
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

# --- Constantes ---
HEADER_FORMAT = '<HBII' 
MAGIC_HEADER = 0xABCD
OPCODE_RECEIVE = 1
OPCODE_SEND = 2

def calys(x, num_rules, params):
    c, s, p, q = params['c'], params['s'], params['p'], params['q']
    rule_outputs = q + np.dot(x, p)
    diff = x[:, None] - c
    exponent = -0.5 * (diff ** 2) / (s ** 2)
    rule_weights = np.exp(exponent).prod(axis=0)
    numerator = np.sum(rule_weights * rule_outputs)
    denominator = np.sum(rule_weights)
    return numerator / (denominator + 1e-8)

def evaluate_local_model(X, y, num_rules, params):
    y_pred = np.array([calys(X[i], num_rules, params) for i in range(len(X))])
    mse = mean_squared_error(y, y_pred)
    y_pred_rounded = np.clip(np.round(y_pred), 1, 3)
    accuracy = accuracy_score(y, y_pred_rounded)
    f1 = f1_score(y, y_pred_rounded, average='weighted', zero_division=0)
    return {'mse': mse, 'accuracy': accuracy, 'f1_score': f1}

def calys_deriv(x, num_rules, params):
    c, s, p, q = params['c'], params['s'], params['p'], params['q']
    rule_outputs = q + np.dot(x, p)
    diff = x[:, None] - c
    exponent = -0.5 * (diff ** 2) / (s ** 2)
    rule_weights = np.exp(exponent).prod(axis=0)
    numerator = np.sum(rule_weights * rule_outputs)
    denominator = np.sum(rule_weights)
    output = numerator / (denominator + 1e-8)
    return output, rule_weights, rule_outputs, denominator

def treinamento_federado(X_train, y_train, params, num_rules, mu, alpha, epochs, shuffle: bool = False, grad_clip: float | None = None):
    num_features = X_train.shape[1]
    c, s, p, q = params['c'].copy(), params['s'].copy(), params['p'].copy(), params['q'].copy()
    c_g, s_g, p_g, q_g = params['c'], params['s'], params['p'], params['q']
    
    for _ in range(epochs):
        if shuffle:
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
        else:
            indices = range(len(X_train))
        for k in indices:
            x, target = X_train[k], y_train[k]
            ys, w, y, b = calys_deriv(x, num_rules, {'c': c, 's': s, 'p': p, 'q': q})
            error = ys - target
            
            dys_dw = (y - ys) / (b + 1e-8)
            dys_dy = w / (b + 1e-8)
            
            diff = x[:, None] - c
            dw_dc = w * diff / (s ** 2)
            dw_ds = w * (diff ** 2) / (s ** 3)
            
            prox_term_c = mu * (c - c_g)
            prox_term_s = mu * (s - s_g)
            prox_term_p = mu * (p - p_g)
            prox_term_q = mu * (q - q_g)

            grad_c = error * dw_dc * dys_dw[None, :] + prox_term_c
            grad_s = error * dw_ds * dys_dw[None, :] + prox_term_s
            grad_p = error * (x[:, None]) * dys_dy[None, :] + prox_term_p
            grad_q = error * dys_dy + prox_term_q

            if grad_clip is not None and grad_clip > 0:
                g = float(grad_clip)
                grad_c = np.clip(grad_c, -g, g)
                grad_s = np.clip(grad_s, -g, g)
                grad_p = np.clip(grad_p, -g, g)
                grad_q = np.clip(grad_q, -g, g)

            c -= alpha * grad_c
            s -= alpha * grad_s
            p -= alpha * grad_p
            q -= alpha * grad_q
    
            c = np.clip(c, 0.0, 1.0)
            s = np.clip(s, 1e-3, 10.0)
            p = np.clip(p, -5.0, 5.0)
            q = np.clip(q, -5.0, 5.0)
            
    return {'c': c, 's': s, 'p': p, 'q': q}

def pack_weights(params):
    return np.concatenate([
        params['c'].flatten(),
        params['s'].flatten(),
        params['p'].flatten(),
        params['q'].flatten()
    ]).astype(np.float32)

def main(client_id, host, port, alpha: float = 0.01, epochs: int = 2, shuffle: bool = False, grad_clip: float | None = None):
    orig_alpha = alpha
    alpha = float(max(1e-8, abs(alpha)))
    if orig_alpha != alpha:
        print(f"[Aviso] Alpha inválido ({orig_alpha}); usando |alpha|={alpha}")
    print(f"Cliente {client_id} iniciando... (alpha={alpha}, epochs={epochs}, shuffle={shuffle}, grad_clip={grad_clip})")
    data = pd.read_csv('drivers/clustered_data_federated.csv')
    client_data = data[data['client'] == client_id]
    
    features = FEATURES
    X_train = normalize_df(client_data[features])
    y_train = client_data['cluster_id'].values
    num_features = X_train.shape[1]
    num_rules = 10
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect((host, port))
            sock.sendall(struct.pack('<I', len(client_id.encode())) + client_id.encode())
            sock.sendall(struct.pack('<I', len(X_train)))
            print(f"Cliente {client_id} conectado.")

            while True:
                header_bytes = sock.recv(struct.calcsize('<HBII'))
                if not header_bytes: break
                
                magic, opcode, weights_len, _ = struct.unpack('<HBII', header_bytes)
                if magic != MAGIC_HEADER or opcode != OPCODE_RECEIVE: continue
                
                mu_bytes = sock.recv(4); mu, = struct.unpack('<f', mu_bytes)
                weights_bytes = sock.recv(weights_len)
                flag = weights_bytes[0]
                offset = 1
                scaler = None
                if flag == 1:
                    num_features = X_train.shape[1]
                    num_f = num_features
                    scaler_min = np.frombuffer(weights_bytes[offset:offset + num_f*4], dtype=np.float32).copy(); offset += num_f*4
                    scaler_scale = np.frombuffer(weights_bytes[offset:offset + num_f*4], dtype=np.float32).copy(); offset += num_f*4
                    class _Scaler: pass
                    scaler = _Scaler(); scaler.data_min_ = scaler_min; scaler.scale_ = scaler_scale

                weights_flat = np.frombuffer(weights_bytes[offset:], dtype=np.float32).copy()
                shapes = ((num_features, num_rules), (num_features, num_rules), (num_features, num_rules), (num_rules,))
                c_s, s_s, p_s = np.prod(shapes[0]), np.prod(shapes[1]), np.prod(shapes[2])
                base_len = c_s + s_s + p_s + shapes[3][0]
                centroid_len = num_features * 3  

                global_params = {}
                global_params['c'] = weights_flat[:c_s].reshape(shapes[0])
                global_params['s'] = weights_flat[c_s:c_s+s_s].reshape(shapes[1])
                global_params['p'] = weights_flat[c_s+s_s:c_s+s_s+p_s].reshape(shapes[2])
                global_params['q'] = weights_flat[c_s+s_s+p_s:base_len]
                if weights_flat.size >= base_len + centroid_len:
                    global_params['centroids'] = weights_flat[base_len:base_len+centroid_len].reshape((3, num_features))
                else:
                    global_params['centroids'] = None
                
                local_params = treinamento_federado(X_train, y_train, global_params, num_rules, mu, alpha, epochs, shuffle=shuffle, grad_clip=grad_clip)
                train_metrics = evaluate_local_model(X_train, y_train, num_rules, local_params)
                centroid_info = None
                if global_params.get('centroids') is not None:
                    cents = global_params['centroids']
                    client_orig = client_data[features].values
                    if scaler is not None:
                        X_for_assign = (client_orig - scaler.data_min_) * scaler.scale_
                    else:
                        X_for_assign = client_orig

                    sums = np.zeros((cents.shape[0], X_for_assign.shape[1]))
                    counts = np.zeros(cents.shape[0], dtype=int)
                    sse = 0.0
                    for i in range(len(X_for_assign)):
                        x = X_for_assign[i]
                        dists = np.linalg.norm(x - cents, axis=1)
                        idx = int(np.argmin(dists))
                        counts[idx] += 1
                        sums[idx] += x
                        sse += dists[idx] ** 2
                    mse_centroid = sse / len(X_for_assign) if len(X_for_assign) > 0 else 0.0
                    centroid_info = {'counts': counts.tolist(), 'sums': sums.tolist(), 'sse': float(sse), 'mse': float(mse_centroid)}

                metrics_with_centroid = train_metrics.copy()
                if centroid_info is not None:
                    metrics_with_centroid['centroid_info'] = centroid_info

                local_weights_bytes = pack_weights(local_params).tobytes()
                metrics_bytes = json.dumps(metrics_with_centroid).encode('utf-8')
                
                response_header = struct.pack('<HBII', MAGIC_HEADER, OPCODE_SEND, len(local_weights_bytes), len(metrics_bytes))
                sock.sendall(response_header + local_weights_bytes + metrics_bytes)
                print(f"Cliente {client_id}: Treino concluído, métricas e pesos enviados.")

        except Exception as e: print(f"Cliente {client_id}: Erro de conexão: {e}")
        finally: print(f"Cliente {client_id}: Conexão encerrada.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--alpha', type=float, default=0.01, help='Client learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='Local epochs per round')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle samples each epoch (off by default)')
    parser.add_argument('--grad-clip', type=float, default=None, help='Element-wise gradient clipping value (None to disable)')
    args = parser.parse_args()
    main(args.id, '127.0.0.1', 9000, alpha=args.alpha, epochs=args.epochs, shuffle=args.shuffle, grad_clip=args.grad_clip)
