import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os

PRINT_PAPER = False

# 1. Carregar dados do FEM
fem_file = "npz/fem_results_2d.json"
if os.path.exists(fem_file):
    with open(fem_file, "r") as f:
        fem_data = json.load(f)
    fem_err = np.array(fem_data["rel_error"])
    fem_time = np.array(fem_data["solve_time"])
else:
    print("Arquivo 'fem_results_2d.json' não encontrado. O plot do FEM será omitido.")
    fem_err, fem_time = [], []

# Dicionários separados para não misturar os dados
pinn_data = {}
pinn_data_paper = {}

# 2. Carregar e organizar dados das PINNs
pinn_files = glob.glob("npz/results_poisson_2d/dados_pinn_2d_*.json")
for file in pinn_files:
    with open(file, 'r') as f:
        data = json.load(f)
        
    err = float(data['error_relativo_medio'])
    train_t = float(data['time_training']) 
    eval_t = float(data['time_evaluation'])
    
    arch = data['architecture']
    width = arch[1] if len(arch) > 2 else arch[0]
    
    if width not in pinn_data:
        pinn_data[width] = {'err': [], 'train_time': [], 'eval_time': []}
        
    pinn_data[width]['err'].append(err)
    pinn_data[width]['train_time'].append(train_t)
    pinn_data[width]['eval_time'].append(eval_t)

# 3. Carregar dados estruturados do seu novo arquivo JSON
eval_file = "PINNs_2D_evaluation.json"  # Altere para PINNs_evaluation_2.json se for o caso
if os.path.exists(eval_file):
    with open(eval_file, "r") as f:
        eval_data = json.load(f)
    
    arch_dict = eval_data.get("arch", {})
    l2_rel_dict = eval_data.get("l2_rel", {})
    times_total_dict = eval_data.get("times_total", {})
    times_eval_dict = eval_data.get("times_eval", {})
    
    sorted_keys = sorted(l2_rel_dict.keys(), key=lambda x: int(x))
    
    for key in sorted_keys:
        width = arch_dict[key][0]
        err = l2_rel_dict[key]
        t_total = times_total_dict[key]
        t_eval = times_eval_dict[key]
        
        if width not in pinn_data_paper:
            pinn_data_paper[width] = {'err': [], 'train_time': [], 'eval_time': []}
            
        pinn_data_paper[width]['err'].append(err)
        pinn_data_paper[width]['train_time'].append(t_total)
        pinn_data_paper[width]['eval_time'].append(t_eval)
else:
    print(f"Arquivo '{eval_file}' não encontrado.")

# 4. Criar a Figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Paleta de cores para identificar a largura da rede
color_map = {
    20: '#2ca02c',  # Verde
    60: '#d62728',  # Vermelho
    120: '#9467bd'  # Roxo
}

# =====================================================================
# ---- Gráfico 4a: Tempo de Solução vs Erro Relativo ----
# =====================================================================
if len(fem_err) > 0:
    ax1.loglog(fem_err, fem_time, linestyle='-', color='steelblue', label='FEM', linewidth=2)

# Plotar dados
for width, metrics in sorted(pinn_data.items()):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    time_sorted = np.array(metrics['train_time'])[sort_indices]
    
    c = color_map.get(width, 'gray')
    ax1.loglog(err_sorted, time_sorted, linestyle='--', color=c, alpha=0.9, 
               label=f'{width} nodes PINNs')
    ax1.loglog(err_sorted, time_sorted, marker='o', linestyle='none', color=c, alpha=0.9)

if PRINT_PAPER:
    # Plotar NOVOS dados do JSON em destaque
    for width, metrics in sorted(pinn_data_paper.items()):
        sort_indices = np.argsort(metrics['err'])
        err_sorted = np.array(metrics['err'])[sort_indices]
        time_sorted = np.array(metrics['train_time'])[sort_indices]
        
        c = color_map.get(width, 'black') # Fallback preto se a rede for diferente
        # Usando marcadores grandes (Diamante), linha contínua grossa e borda preta
        ax1.loglog(err_sorted, time_sorted, linestyle='-', color=c, linewidth=1.5,
                label=f'{width} nodes PINNs (Paper)')
        ax1.loglog(err_sorted, time_sorted, marker='D', markeredgecolor='black', 
                linestyle='none', color=c, zorder=5)

ax1.set_xlabel('Relative Error')
ax1.set_ylabel('Total time to solve in sec')
ax1.set_title(r'(a) Plot of time to solve FEM and train PINN in sec versus $\ell^2$ relative error.')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()


# =====================================================================
# ---- Gráfico 4b: Tempo de Avaliação vs Erro Relativo ----
# =====================================================================
if len(fem_err) > 0:
    ax2.loglog(fem_err, fem_time, linestyle='-', color='#8c564b', label='FEM solving time', linewidth=2)

# Plotar dados
for width, metrics in sorted(pinn_data.items()):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    eval_time_sorted = np.array(metrics['eval_time'])[sort_indices]
    
    c = color_map.get(width, 'gray')
    ax2.loglog(err_sorted, eval_time_sorted, linestyle='--', color=c, alpha=0.9, 
               label=f'{width} nodes PINNs')
    ax2.loglog(err_sorted, eval_time_sorted, marker='o', linestyle='none', color=c, alpha=0.9)

if PRINT_PAPER:
    # Plotar NOVOS dados do JSON em destaque
    for width, metrics in sorted(pinn_data_paper.items()):
        sort_indices = np.argsort(metrics['err'])
        err_sorted = np.array(metrics['err'])[sort_indices]
        eval_time_sorted = np.array(metrics['eval_time'])[sort_indices]
        
        c = color_map.get(width, 'black')
        ax2.loglog(err_sorted, eval_time_sorted, linestyle='-', color=c, linewidth=1.5,
                label=f'{width} nodes PINNs (Paper)')
        ax2.loglog(err_sorted, eval_time_sorted, marker='D', markeredgecolor='black', 
                linestyle='none', color=c, zorder=5)

ax2.set_xlabel('Relative Error')
ax2.set_ylabel('Time in sec')
ax2.set_title('(b) Plot of time to interpolate FEM and evaluate PINN in sec versus relative error.')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

# plt.suptitle('Figure 4: Plot for 2D Poisson equation of time in sec versus $\ell^2$ relative error.', fontsize=14, y=0.0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

plt.savefig("POISSON-2D-FEM-PINNs.png", dpi=150)
plt.show()