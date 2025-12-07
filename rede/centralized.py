import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
import os
import argparse
import io, zipfile
from typing import Optional, Dict

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

def get_scaler_like():
    class _Scaler: pass
    s = _Scaler()
    s.data_min_ = MIN_VALUES.astype(np.float32)
    s.scale_ = SCALE_VALUES.astype(np.float32)
    return s

# --- Parameter loader (inlined) ---
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
                print("One or more parameters missing in the .npz file.")
                return None
            c = np.asarray(c); s = np.asarray(s); p = np.asarray(p); q = np.asarray(q)
            if c.size != num_features * num_rules or s.size != num_features * num_rules or p.size != num_features * num_rules or q.size != num_rules:
                print("Parameter shapes do not match expected dimensions.")
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
    # paths += [
    #     os.path.join('rede', 'parametros_globais'),
    #     os.path.join('rede', 'resultados', 'FedAvg', 'final_params_FedAvg.npz'),
    #     os.path.join(os.path.dirname(__file__), '..', 'rede.zip'),
    #     os.path.join(os.path.dirname(__file__), 'rede.zip'),
    # ]
    for p in paths:
        # if os.path.isdir(p):
        #     loaded = _load_from_dir(p, num_features, num_rules)
        #     if loaded: return   
        if os.path.isfile(p):
            print(f"Trying to load parameters from: {p}")
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
    output = numerator / (denominator + 1e-8)
    return output

def evaluate_model(X, y, num_rules, c, s, p, q):
    y_pred = np.array([calys(X[i], X.shape[1], num_rules, c, s, p, q) for i in range(len(X))])
    mse = mean_squared_error(y, y_pred)
    y_pred_rounded = np.clip(np.round(y_pred), 1, 3)
    
    accuracy = accuracy_score(y, y_pred_rounded)
    precision = precision_score(y, y_pred_rounded, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred_rounded, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred_rounded, average='weighted', zero_division=0)
    
    metrics = {'mse': mse, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
    return metrics

def treinamento(X_train, y_train, X_test, y_test, num_rules, alpha=0.001, max_epochs=10, init_params=None):
    num_samples, num_features = X_train.shape
    
    if init_params is None:
        c = np.random.rand(num_features, num_rules)
        s = np.random.rand(num_features, num_rules)
        p = np.random.randn(num_features, num_rules) * 0.1
        q = np.random.randn(num_rules) * 0.1
    else:
        c = init_params['c'].copy()
        s = init_params['s'].copy()
        p = init_params['p'].copy()
        q = init_params['q'].copy()
    
    history = {
        'train_mse': [], 'train_accuracy': [], 'train_f1_score': [], 'train_precision': [], 'train_recall': [],
        'val_mse': [], 'val_accuracy': [], 'val_f1_score': [], 'val_precision': [], 'val_recall': []
    }
    
    print("Iniciando treinamento centralizado...")
    train_metrics = evaluate_model(X_train, y_train, num_rules, c, s, p, q)
    val_metrics = evaluate_model(X_test, y_test, num_rules, c, s, p, q)
    for metric in train_metrics:
        history[f'train_{metric}'].append(train_metrics[metric])
        history[f'val_{metric}'].append(val_metrics[metric])

    for epoch in range(max_epochs):
        indices = np.random.permutation(num_samples)
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

        for k in range(num_samples):
            x, target_val = X_train_shuffled[k], y_train_shuffled[k]
            ys, w, y, b = calys_deriv(x, num_features, num_rules, c, s, p, q)
            error = ys - target_val
            diff_deriv = x[:, None] - c
            c -= alpha * error * (w * diff_deriv / (s ** 2)) * ((y - ys) / (b + 1e-8))[None, :]
            s -= alpha * error * (w * (diff_deriv ** 2) / (s ** 3)) * ((y - ys) / (b + 1e-8))[None, :]
            p -= alpha * error * (x[:, None]) * (w / (b + 1e-8))[None, :]
            q -= alpha * error * (w / (b + 1e-8))

        train_metrics = evaluate_model(X_train, y_train, num_rules, c, s, p, q)
        val_metrics = evaluate_model(X_test, y_test, num_rules, c, s, p, q)

        for metric in train_metrics:
            history[f'train_{metric}'].append(train_metrics[metric])
            history[f'val_{metric}'].append(val_metrics[metric])

        print(f"Epoch {epoch + 1}/{max_epochs} | Train MSE: {train_metrics['mse']:.4f} | Val MSE: {val_metrics['mse']:.4f} | Val Acc: {train_metrics['accuracy']:.4f}")

    return c, s, p, q, history

def calys_deriv(x, num_inputs, num_rules, centers, sigmas, weights, biases):
    rule_outputs = biases + np.dot(x, weights)
    diff = x[:, None] - centers
    exponent = -0.5 * (diff ** 2) / (sigmas ** 2)
    rule_weights = np.exp(exponent).prod(axis=0)
    numerator = np.sum(rule_weights * rule_outputs)
    denominator = np.sum(rule_weights)
    output = numerator / (denominator + 1e-8)
    return output, rule_weights, rule_outputs, denominator

def classify_input(X, strat:str):
    x = normalize_df(X[FEATURES])
    try:
        params = load_initial_params(num_features=x.shape[1], num_rules=10, prefer_paths=[os.path.join(
            'rede','resultados', strat, f"final_params_{strat}.npz")])
        c, s, p, q = params['c'], params['s'], params['p'], params['q']
    except Exception as e:
        raise ValueError(f"Falha ao carregar os parâmetros do modelo: {e}")
    
    output = np.array([calys(x[0], x.shape[1], 10, c, s, p, q)])
    print(np.clip(output, 0, 3))
    return np.clip(output, 0, 3)

def main(init_mode: str = 'random', epochs: int = 5):
    print("--- INICIANDO TREINAMENTO CENTRALIZADO ---")
    data_path = 'drivers/clustered_data.csv'
    if not os.path.exists(data_path):
        print(f"Erro: Arquivo de dados não encontrado em '{data_path}'"); return
    
    data = pd.read_csv(data_path)
    features = FEATURES
    target = 'cluster_id' 

    X = normalize_df(data[features]) #
    y = data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    init = None
    if init_mode == 'load':
        init = load_initial_params(num_features=X_train.shape[1], num_rules=10)
        if init is not None:
            print('Initializing ANFIS parameters from provided files (init=load).')
        else:
            print('No compatible parameter files found; falling back to random initialization.')
    else:
        print('Initializing ANFIS parameters randomly (init=random).')
    # epochs configurable via CLI
    print(f"Parâmetros: epochs={epochs}")
    c, s, p, q, history = treinamento(X_train, y_train, X_test, y_test, num_rules=10, max_epochs=int(epochs), init_params=init)

    history_df = pd.DataFrame(history)
    history_df.to_csv('history_centralizado.csv', index=False)
    print("Histórico do treinamento centralizado salvo em 'history_centralizado.csv'")

    final_metrics_list = []
    for d_set in ['train', 'val']:
        final_metrics_list.append({
            'dataset': 'treino' if d_set == 'train' else 'validacao',
            'mse': history[f'{d_set}_mse'][-1],
            'accuracy': history[f'{d_set}_accuracy'][-1],
            'precision': history[f'{d_set}_precision'][-1],
            'recall': history[f'{d_set}_recall'][-1],
            'f1_score': history[f'{d_set}_f1_score'][-1]
        })
        
    metrics_df = pd.DataFrame(final_metrics_list)
    print("\n--- RELATÓRIO FINAL ---")
    print(metrics_df.to_string())
    metrics_df.to_csv('relatorio_centralizado.csv', index=False)
    print("\nRelatório de métricas salvo em 'relatorio_centralizado.csv'")

    labels = sorted(data['cluster_id'].unique())
    central_means_raw = np.vstack([data[data['cluster_id'] == lbl][features].mean().values for lbl in labels])
    central_counts = np.array([len(data[data['cluster_id'] == lbl]) for lbl in labels])
    scaler_like = get_scaler_like()
    central_means_scaled = (central_means_raw - scaler_like.data_min_) * scaler_like.scale_
    os.makedirs(os.path.join('resultados', 'central'), exist_ok=True)
    np.savez_compressed(os.path.join('resultados', 'central', 'final_params_central.npz'),
                        centroids=central_means_scaled, counts=central_counts)
    pd.DataFrame({'cluster': labels, 'count': central_counts}).to_csv(os.path.join('resultados', 'central', 'central_counts.csv'), index=False)
    print("Centroides e contagens centrais salvos em 'resultados/central'")

    print("\n--- TREINAMENTO CENTRALIZADO CONCLUÍDO ---")

    # --- Consolidated chart: F1-score per client (A–J) for centralized model ---
    try:
        if 'driver' in data.columns:
            X_all = normalize_df(data[features])
            y_all = data[target].values
            num_rules = 10
            y_pred = np.array([calys(X_all[i], X_all.shape[1], num_rules, c, s, p, q) for i in range(len(X_all))])
            y_pred_rounded = np.clip(np.round(y_pred), 1, 3)

            desired_clients = [chr(ord('A') + i) for i in range(10)]
            f1_by_client = {}
            for client_id in desired_clients:
                mask = (data['driver'] == client_id) 
                if mask.any():
                    f1_val = f1_score(y_all[mask], y_pred_rounded[mask], average='weighted', zero_division=0)
                    f1_by_client[client_id] = float(f1_val)
                else:
                    f1_by_client[client_id] = np.nan

            os.makedirs('graficos', exist_ok=True)
            labels = list(f1_by_client.keys())
            values = [f1_by_client[k] for k in labels]
            plt.figure(figsize=(8, 5))
            bars = plt.bar(labels, values, color='#4C78A8')
            plt.ylim(0, 1)
            plt.xlabel('Client')
            plt.ylabel('F1-score (weighted)')
            plt.title('F1-score per client (A–J) - Centralized')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            for b, v in zip(bars, values):
                txt = 'N/A' if np.isnan(v) else f"{v:.3f}"
                plt.text(b.get_x() + b.get_width()/2, (0 if np.isnan(v) else v) + 0.02, txt,
                         ha='center', va='bottom', fontsize=9)
            out_png = os.path.join('graficos', 'f1_clients_A_J_Centralized.png')
            plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
            print(f"Consolidated F1 (A–J) chart for Centralized saved at '{out_png}'")
        else:
            print("Column 'driver' not found in drivers/clustered_data.csv; cannot plot centralized F1 per client.")
    except Exception as e:
        print(f"[Warning] Failed to generate centralized consolidated F1 chart: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', choices=['random', 'load'], default='random', help='How to initialize model parameters')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs for centralized training')
    args = parser.parse_args()
    main(init_mode=args.init, epochs=args.epochs)
