import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os

# 1. Carregar dados do FEM
fem_file = "npz/fem_results.json"
if os.path.exists(fem_file):
    with open(fem_file, "r") as f:
        fem_data = json.load(f)
    fem_err = np.array(fem_data["rel_error"])
    fem_time = np.array(fem_data["solve_time"])
else:
    print("Arquivo 'fem_results.json' não encontrado. Rode o script do FEM primeiro.")
    fem_err, fem_time = [], []

# 2. Carregar e organizar dados das PINNs
pinn_files = glob.glob("npz/pinn_lbfgs_*.npz")

# Dicionário para agrupar dados por largura (número de nós)
pinn_data_grouped = {}

for file in pinn_files:
    data = np.load(file, allow_pickle=True)
    err = float(data['error_relativo'])
    train_t = float(data['time_training'])
    eval_t = float(data['time_evaluation'])
    
    # Extrair largura da rede
    arch = data['architecture'].tolist()
    width = arch[1] if len(arch) > 1 else "Unknown"
    
    # Adicionar no grupo correspondente
    if width not in pinn_data_grouped:
        pinn_data_grouped[width] = {'err': [], 'train_time': [], 'eval_time': []}
        
    pinn_data_grouped[width]['err'].append(err)
    pinn_data_grouped[width]['train_time'].append(train_t)
    pinn_data_grouped[width]['eval_time'].append(eval_t)

# 3. Criar a Figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Paleta de cores para as diferentes PINNs
colors = ['#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#e377c2']

# ---- Gráfico 4a: Tempo de Solução vs Erro Relativo ----
if len(fem_err) > 0:
    ax1.loglog(fem_err, fem_time, linestyle='-', color='steelblue', label='FEM')

# Plotar cada grupo de PINN
for i, (width, metrics) in enumerate(sorted(pinn_data_grouped.items())):
    # Ordenar os pontos pelo erro para que a linha não cruze sobre si mesma
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    time_sorted = np.array(metrics['train_time'])[sort_indices]
    
    c = colors[i % len(colors)]
    ax1.loglog(err_sorted, time_sorted, marker='o', linestyle='-', color=c, 
               markersize=5, label=f'{width} nodes PINNs')

ax1.set_xlabel(r'Relative $\ell^2$ Error')
ax1.set_ylabel('Total time to solve in sec')
ax1.set_title('(a) Plot of time to solve FEM and train PINN\nversus $\ell^2$ relative error.')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# ---- Gráfico 4b: Tempo de Avaliação vs Erro Relativo ----
if len(fem_err) > 0:
    ax2.loglog(fem_err, fem_time, linestyle='-', color='rosybrown', label='FEM solving time')

for i, (width, metrics) in enumerate(sorted(pinn_data_grouped.items())):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    eval_time_sorted = np.array(metrics['eval_time'])[sort_indices]
    
    c = colors[i % len(colors)]
    ax2.loglog(err_sorted, eval_time_sorted, marker='o', linestyle='-', color=c, 
               markersize=5, label=f'{width} nodes PINNs')

ax2.set_xlabel(r'Relative $\ell^2$ Error')
ax2.set_ylabel('Time in sec')
ax2.set_title('(b) Plot of time to interpolate FEM and evaluate PINN\nin sec versus relative error.')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.suptitle('Figure 4: Plot for 2D Poisson equation of time in sec versus $\ell^2$ relative error.', fontsize=14)
plt.tight_layout()
plt.savefig("fem_vs_pinn.png", dpi=150)
plt.show()