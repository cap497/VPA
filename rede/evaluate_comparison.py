import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Use a Matplotlib style similar to seaborn's whitegrid without requiring seaborn
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('default')
plt.rcParams.update({'font.size': 10})
BASE = '.'
GRAFICOS_DIR = os.path.join(BASE, 'graficos')
RESULTS_DIR = os.path.join(BASE, 'resultados')

def _discover_strategies():
    strategies = []
    if not os.path.isdir(RESULTS_DIR):
        return strategies
    for name in os.listdir(RESULTS_DIR):
        sub = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(sub):
            continue
        report_path = os.path.join(sub, 'anfis', f'relatorio_{name}.csv')
        if os.path.exists(report_path):
            strategies.append(name)
    return strategies

def plot_comparison_evolution():
    central_hist_path = os.path.join(BASE, 'history_centralizado.csv')
    if not os.path.exists(central_hist_path):
        print(f"Warning: '{central_hist_path}' not found. Skipping evolution plots.")
        return
    central_hist = pd.read_csv(central_hist_path)

    central_hist_evol = central_hist.reset_index(drop=True)
    central_x = np.arange(0, len(central_hist_evol))

    fed_histories = {}
    strategies = [s for s in _discover_strategies() if s in ('FedAvg', 'FedProx')]
    for strat in strategies:
        path = os.path.join(RESULTS_DIR, strat, 'anfis', f'relatorio_{strat}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            fed_histories[strat] = df.reset_index(drop=True)
    if not fed_histories:
        print("Warning: no federated reports found. Skipping evolution plots.")
        return
    print(f"Including strategies in evolution plot: {', '.join(fed_histories.keys())}")

    styles = {
        'Centralized': dict(color='b', linestyle='-', marker=None),
        'FedAvg': dict(color='r', linestyle='--', marker=None),
        'FedProx': dict(color='g', linestyle='-.', marker=None),
    }
    # Build fallback styles from Matplotlib's default color cycle
    fallback_colors = plt.rcParams.get('axes.prop_cycle', None)
    if fallback_colors is not None:
        fallback_colors = fallback_colors.by_key().get('color', [])
    if not fallback_colors:
        fallback_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fallback_styles = [dict(color=fallback_colors[i % len(fallback_colors)], linestyle='--', marker=None)
                       for i in range(max(3, len(fed_histories)))]
    plt.figure(figsize=(8, 5))
    plt.plot(central_x, central_hist_evol['val_accuracy'], label='Centralized', **styles['Centralized'])
    min_accs = [central_hist_evol['val_accuracy'].min()]
    for i, (strat, hist) in enumerate(fed_histories.items()):
        style = styles.get(strat, fallback_styles[i % len(fallback_styles)])
        plt.plot(hist['round'], hist['val_accuracy'], label=f'Federated ({strat})', **style)
        if 'val_accuracy' in hist:
            min_accs.append(hist['val_accuracy'].min())
    plt.xlabel('Epoch / Round')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.ylim(bottom=max(0, min(min_accs) - 0.05))
    os.makedirs(GRAFICOS_DIR, exist_ok=True)
    acc_eps = os.path.join(GRAFICOS_DIR, 'evolution_val_accuracy.eps')
    acc_png = os.path.join(GRAFICOS_DIR, 'evolution_val_accuracy.png')
    plt.savefig(acc_eps, format='eps')
    plt.savefig(acc_png, dpi=300)
    plt.close()
    print(f"Saved accuracy evolution to '{acc_eps}' and '{acc_png}'")

    plt.figure(figsize=(8, 5))
    plt.plot(central_x, central_hist_evol['val_mse'], label='Centralized', **styles['Centralized'])
    for i, (strat, hist) in enumerate(fed_histories.items()):
        style = styles.get(strat, fallback_styles[i % len(fallback_styles)])
        plt.plot(hist['round'], hist['val_mse'], label=f'Federated ({strat})', **style)
    plt.xlabel('Epoch / Round')
    plt.ylabel('Validation MSE')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.ylim(bottom=0)
    mse_eps = os.path.join(GRAFICOS_DIR, 'evolution_val_mse.eps')
    mse_png = os.path.join(GRAFICOS_DIR, 'evolution_val_mse.png')
    plt.savefig(mse_eps, format='eps')
    plt.savefig(mse_png, dpi=300)
    plt.close()
    print(f"Saved MSE evolution to '{mse_eps}' and '{mse_png}'")
def plot_final_metrics_comparison():
    central_report_path = os.path.join(BASE, 'relatorio_centralizado.csv')
    if not os.path.exists(central_report_path):
        print(f"Warning: '{central_report_path}' not found. Skipping final metrics plot.")
        return
    central_metrics = pd.read_csv(central_report_path)
    central_final = central_metrics[central_metrics['dataset'] == 'validacao'].iloc[0]
    rows = [('Centralized', central_final['accuracy'], central_final['f1_score'], central_final['mse'])]
    for strat in [s for s in _discover_strategies() if s in ('FedAvg', 'FedProx')]:
        fed_report_path = os.path.join(RESULTS_DIR, strat, 'anfis', f'relatorio_{strat}.csv')
        if os.path.exists(fed_report_path):
            fed_metrics = pd.read_csv(fed_report_path)
            fed_final = fed_metrics.iloc[-1]
            rows.append((f'Federated ({strat})', fed_final['val_accuracy'], fed_final['val_f1_score'], fed_final['val_mse']))
    if len(rows) <= 1:
        print("Warning: no federated reports found. Skipping final metrics plot.")
        return
    df_plot = pd.DataFrame(rows, columns=['Method', 'Accuracy', 'F1-Score', 'MSE']).set_index('Method')
    ax = df_plot.plot(kind='bar', figsize=(10, 6), rot=0)
    ax.set_ylabel('Score')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    os.makedirs(GRAFICOS_DIR, exist_ok=True)
    metrics_eps = os.path.join(GRAFICOS_DIR, 'final_metrics_comparison.eps')
    metrics_png = os.path.join(GRAFICOS_DIR, 'final_metrics_comparison.png')
    plt.savefig(metrics_eps, format='eps')
    plt.savefig(metrics_png, dpi=300)
    plt.close()
    print(f"Saved final metrics comparison to '{metrics_eps}' and '{metrics_png}'")
def compare():
    os.makedirs(GRAFICOS_DIR, exist_ok=True)
    plot_comparison_evolution()
    plot_final_metrics_comparison()
    print(f"\nComparison process completed. Figures saved in '{GRAFICOS_DIR}'.")
if __name__ == '__main__':
    compare()
