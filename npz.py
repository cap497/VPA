import numpy as np
import os

# Carrega o arquivo
data = np.load(os.path.join('rede', 'resultados', 'central', 'final_params_central.npz'))

# Lista todas as chaves (nomes dos arrays salvos)
print("Chaves no arquivo:", data.files)

# Acessa cada array pelo nome
for key in data.files:
    print(f"{key} -> shape={data[key].shape}, dtype={data[key].dtype}")
    print(data[key][:10])  # mostra os 10 primeiros valores