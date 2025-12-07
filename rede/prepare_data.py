import pandas as pd
import numpy as np
from pathlib import Path

# Inlined FEATURES to remove dependency on fixed_normalization
FEATURES = [
    'speed',
    'acc_norm',
    'engine_speed',
    'throttle_position',
    'delta_acc_lat',
]


def prepare_federated_data(input_path='drivers/clustered_data.csv', output_path='drivers/clustered_data_federated.csv', num_clients=10):
    try:
        data = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_path}' não foi encontrado.")
        return

    client_ids = [chr(ord('A') + i) for i in range(num_clients)]
    
    data_size = len(data)
    assignments = np.tile(client_ids, (data_size // num_clients) + 1)[:data_size]
    
    data['client'] = np.random.permutation(assignments)
    
    data.to_csv(output_path, index=False)
    print(f"Dados preparados e salvos em '{output_path}' com a coluna 'client'.")
    print("Distribuição de dados por cliente:")
    print(data['client'].value_counts().sort_index())


def build_from_analyzed_csv(analyzed_csv_path: str, out_clustered_path='drivers/clustered_data.csv', out_federated_path='drivers/clustered_data_federated.csv'):
    """
    Build the clustered_data.csv and clustered_data_federated.csv from the provided
    analyzed_data_drivers.csv shared on Drive.

    Expected columns in analyzed file: at least
      - 'driver' (A..J), 'cluster_id', and FEATURE columns.

    We'll subset to FEATURES + 'cluster_id' and map 'driver' to 'client'.
    """
    analyzed_csv_path = Path(analyzed_csv_path)
    if not analyzed_csv_path.exists():
        print(f"Erro: '{analyzed_csv_path}' não encontrado. Informe o caminho correto do analyzed_data_drivers.csv")
        return

    df = pd.read_csv(analyzed_csv_path)

    missing = [c for c in FEATURES + ['cluster_id', 'driver'] if c not in df.columns]
    if missing:
        print(f"Aviso: Colunas ausentes na base analisada: {missing}")
    cols = [c for c in FEATURES + ['cluster_id', 'driver'] if c in df.columns]
    data = df[cols].copy()
    if 'cluster_id' in data.columns:
        data['cluster_id'] = pd.to_numeric(data['cluster_id'], errors='coerce').fillna(1).astype(int)
        data['cluster_id'] = data['cluster_id'].clip(lower=1, upper=3)
    if 'driver' in data.columns:
        data['client'] = data['driver']
    else:
        n = len(data)
        client_ids = sorted(list(set(df.get('driver', list('ABCDEFGHIJ')))))
        assign = np.tile(client_ids, (n // len(client_ids)) + 1)[:n]
        data['client'] = assign

    Path(out_clustered_path).parent.mkdir(parents=True, exist_ok=True)
    data.drop(columns=['client'], errors='ignore').to_csv(out_clustered_path, index=False)
    data.to_csv(out_federated_path, index=False)
    print(f"Bases geradas:\n - {out_clustered_path}\n - {out_federated_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare federated datasets from analyzed CSV or fallback to synthetic split.')
    parser.add_argument('--source', type=str, default=None, help='Path to analyzed_data_drivers.csv')
    parser.add_argument('--clients', type=int, default=10, help='Number of clients for synthetic split fallback')
    args = parser.parse_args()

    if args.source:
        build_from_analyzed_csv(args.source)
    else:
        default_analyzed = str(Path.home() / 'Downloads' / 'analyzed_data_drivers.csv')
        if Path(default_analyzed).exists():
            build_from_analyzed_csv(default_analyzed)
        else:
            print("Arquivo analyzed_data_drivers.csv não encontrado; usando preparação padrão com divisão sintética.")
            prepare_federated_data(num_clients=args.clients)
