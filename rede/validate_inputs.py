import pandas as pd
import numpy as np
from pathlib import Path

# Inlined normalization constants to remove dependency on fixed_normalization
FEATURES = [
    'speed',
    'acc_norm',
    'engine_speed',
    'throttle_position',
    'delta_acc_lat',
]
MIN_VALUES = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
MAX_VALUES = np.array([120.0, 5.0, 10000.0, 100.0, 3.0], dtype=np.float32)

EXPECTED_FEATURE_BOUNDS = dict(zip(FEATURES, zip(MIN_VALUES, MAX_VALUES)))


def summarize_bounds(df: pd.DataFrame):
    summary = []
    for feat in FEATURES:
        if feat not in df.columns:
            summary.append({
                'feature': feat,
                'present': False,
                'min': None,
                'max': None,
                'below_min': None,
                'above_max': None,
            })
            continue
        x = df[feat].astype(float)
        lo, hi = EXPECTED_FEATURE_BOUNDS[feat]
        below = int((x < lo).sum())
        above = int((x > hi).sum())
        summary.append({
            'feature': feat,
            'present': True,
            'min': float(np.nanmin(x)),
            'max': float(np.nanmax(x)),
            'below_min': below,
            'above_max': above,
        })
    return pd.DataFrame(summary)


def validate_clusters(df: pd.DataFrame):
    if 'cluster_id' not in df.columns:
        return {'present': False, 'unique': [], 'invalid': []}
    c = pd.to_numeric(df['cluster_id'], errors='coerce')
    uniq = sorted([int(u) for u in pd.unique(c.dropna())])
    invalid = [int(u) for u in uniq if u < 1 or u > 3]
    return {'present': True, 'unique': uniq, 'invalid': invalid}


def main():
    clustered = Path('drivers/clustered_data.csv')
    federated = Path('drivers/clustered_data_federated.csv')
    for p in [clustered, federated]:
        if not p.exists():
            print(f"Arquivo não encontrado: {p}")
            continue
        print(f"\n=== Verificando: {p} ===")
        df = pd.read_csv(p)
        b = summarize_bounds(df)
        print("\nFaixas por feature (valores fora do intervalo serão clipados na normalização):")
        print(b.to_string(index=False))
        cl = validate_clusters(df)
        if not cl['present']:
            print("\nColuna 'cluster_id' ausente.")
        else:
            print(f"\nClusters únicos encontrados: {cl['unique']}")
            if cl['invalid']:
                print(f"Valores de cluster fora de [1..3]: {cl['invalid']}")
            else:
                print("Clusters padronizados para [1..3].")


if __name__ == '__main__':
    main()
