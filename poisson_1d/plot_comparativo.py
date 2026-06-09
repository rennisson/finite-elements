import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os

# 1. Carregar dados do FEM
# Assumindo que você criou um json similar para o caso 1D
fem_file = "fem_results_1d.json" 
if os.path.exists(fem_file):
    with open(fem_file, "r") as f:
        fem_data = json.load(f)
    fem_err = np.array(fem_data["rel_error"])
    fem_time = np.array(fem_data["solve_time"])
else:
    print(f"Arquivo '{fem_file}' não encontrado. O plot do FEM será omitido.")
    fem_err, fem_time = [], []

# 2. Carregar e organizar dados das PINNs
# Utilizamos o padrão definido no código pinn_poisson_1d.py
pinn_files = glob.glob("npz/dados_pinn_1d_*.npz")

# Dicionário para agrupar dados por número de camadas
# Formato: {layers: {'err': [], 'train_time': [], 'eval_time': []}}
pinn_data_grouped = {}

for file in pinn_files:
    data = np.load(file, allow_pickle=True)
    err = float(data['error_relativo'])
    train_t = float(data['time_training'])
    eval_t = float(data['time_evaluation'])
    
    # Extrair número de camadas a partir dos dados salvos
    if 'num_hidden_layers' in data:
        layers = int(data['num_hidden_layers'])
    else:
        # Fallback caso a chave não exista
        arch = data['architecture'].tolist()
        layers = len(arch) - 2
    
    # Adicionar no grupo correspondente
    if layers not in pinn_data_grouped:
        pinn_data_grouped[layers] = {'err': [], 'train_time': [], 'eval_time': []}
        
    pinn_data_grouped[layers]['err'].append(err)
    pinn_data_grouped[layers]['train_time'].append(train_t)
    pinn_data_grouped[layers]['eval_time'].append(eval_t)

# 3. Criar a Figura semelhante à do artigo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Paleta de cores para as diferentes PINNs (semelhante ao artigo: verde, vermelho, roxo...)
colors = ['#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#e377c2']

# ---- Gráfico 2a: Tempo de Treinamento/Solução vs Erro Relativo ----
if len(fem_err) > 0:
    ax1.loglog(fem_err, fem_time, linestyle='-', color='steelblue', label='FEM')

# Plotar cada grupo de PINN
for i, (layers, metrics) in enumerate(sorted(pinn_data_grouped.items())):
    # Ordenar os pontos pelo erro para que a linha não cruze sobre si mesma
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    time_sorted = np.array(metrics['train_time'])[sort_indices]
    
    c = colors[i % len(colors)]
    ax1.loglog(err_sorted, time_sorted, marker='o', linestyle='-', color=c, 
               markersize=5, label=f'{layers}-layer PINNs')

ax1.set_xlabel(r'Relative $\ell^2$ Error')
ax1.set_ylabel('Total time to solve in sec')
ax1.set_title('(a) Plot of time to solve FEM and train PINN\nversus $\ell^2$ relative error.')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# ---- Gráfico 2b: Tempo de Avaliação vs Erro Relativo ----
if len(fem_err) > 0:
    ax2.loglog(fem_err, fem_time, linestyle='-', color='rosybrown', label='FEM solving time')

for i, (layers, metrics) in enumerate(sorted(pinn_data_grouped.items())):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    eval_time_sorted = np.array(metrics['eval_time'])[sort_indices]
    
    c = colors[i % len(colors)]
    ax2.loglog(err_sorted, eval_time_sorted, marker='o', linestyle='-', color=c, 
               markersize=5, label=f'{layers}-layer PINNs')

ax2.set_xlabel(r'Relative $\ell^2$ Error')
ax2.set_ylabel('Time in sec')
ax2.set_title('(b) Plot of time to interpolate FEM and evaluate PINN\nin sec versus relative error.')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.suptitle('Figure 2: Plot for 1D Poisson equation of time in sec versus $\ell^2$ relative error.', fontsize=14)
plt.tight_layout()

# Garante que a pasta 'results' existe antes de salvar
os.makedirs("results", exist_ok=True)
plt.savefig("results/fem_vs_pinn_1d.png", dpi=150)
plt.show()