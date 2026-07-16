import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os

PRINT_PAPER = True

# 1. Carregar dados do FEM
fem_file = "npz/fem_results_3d.json"
if os.path.exists(fem_file):
    with open(fem_file, "r") as f:
        fem_data = json.load(f)
    fem_err = np.array(fem_data["rel_error"])
    fem_time = np.array(fem_data["solve_time"])
    fem_eval_time = np.array(fem_data.get("eval_time", []))
else:
    print(f"Arquivo '{fem_file}' não encontrado. O plot do FEM será omitido.")
    fem_err, fem_time, fem_eval_time = [], [], []

# Dicionários separados para não misturar os dados
pinn_data = {}
pinn_data_paper = {}

# 2. Carregar e organizar dados das PINNs gerados localmente
# Espera-se que os arquivos gerados tenham prefixo dados_pinn_3d_...
pinn_files = glob.glob("npz/results_poisson_3d/dados_pinn_3d_*.json")
if not pinn_files:
    # Busca alternativa no diretorio atual
    pinn_files = glob.glob("dados_pinn_3d_*.json")

for file in pinn_files:
    with open(file, 'r') as f:
        data = json.load(f)
        
    err = float(data['error_relativo_medio'])
    train_t = float(data['time_training_lbfgs']) # Pega o tempo total incluindo L-BFGS
    eval_t = float(data['time_evaluation'])
    
    arch = data['architecture']
    # A largura na Poisson 3D (dado o input dimension = 3) será o segundo elemento de 'arch'
    width = arch[1] if len(arch) > 2 else arch[0]
    
    if width not in pinn_data:
        pinn_data[width] = {'err': [], 'train_time': [], 'eval_time': []}
        
    pinn_data[width]['err'].append(err)
    pinn_data[width]['train_time'].append(train_t)
    pinn_data[width]['eval_time'].append(eval_t)

# 3. Carregar dados estruturados JSON do paper (se disponível)
eval_file = "PINNs_3D_evaluation.json" 
if os.path.exists(eval_file):
    with open(eval_file, "r") as f:
        eval_data = json.load(f)
    
    arch_dict = eval_data.get("arch", {})
    l2_rel_dict = eval_data.get("l2_rel", {})
    times_total_dict = eval_data.get("times_total", {})
    times_eval_dict = eval_data.get("times_eval", {})
    
    sorted_keys = sorted(l2_rel_dict.keys(), key=lambda x: int(x))
    
    for key in sorted_keys:
        # A largura na estruturação do paper costuma ser o primeiro elemento da lista arch fornecida
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

# Paleta de cores para identificar a largura da rede (20 ou 60 nodes)
color_map = {
    20: '#2ca02c',  # Verde
    60: '#d62728',  # Vermelho
}

# =====================================================================
# ---- Gráfico 6a: Tempo de Solução vs Erro Relativo ----
# =====================================================================
if len(fem_err) > 0:
    ax1.loglog(fem_err, fem_time, linestyle='-', color='steelblue', label='FEM', linewidth=2)

# Plotar dados
for width, metrics in sorted(pinn_data.items()):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    time_sorted = np.array(metrics['train_time'])[sort_indices]
    
    c = color_map.get(width, 'gray')
    ax1.loglog(err_sorted, time_sorted, linestyle='-', color=c, alpha=0.9, 
               label=f'{width} nodes PINNs')
    ax1.loglog(err_sorted, time_sorted, marker='o', linestyle='none', color='darkorange', alpha=1.0, zorder=3)

if PRINT_PAPER:
    # Plotar NOVOS dados do JSON em destaque
    for width, metrics in sorted(pinn_data_paper.items()):
        sort_indices = np.argsort(metrics['err'])
        err_sorted = np.array(metrics['err'])[sort_indices]
        time_sorted = np.array(metrics['train_time'])[sort_indices]
        
        c = color_map.get(width, 'black')
        ax1.loglog(err_sorted, time_sorted, linestyle='-', color=c, linewidth=1.5,
                label=f'{width} nodes PINNs (Paper)')
        ax1.loglog(err_sorted, time_sorted, marker='D', markeredgecolor='black', 
                linestyle='none', color=c, zorder=5)

ax1.set_xlabel('Relative Error')
ax1.set_ylabel('Total time to solve in sec')
ax1.set_title(r'(a) Plot of time to solve FEM and train PINN in sec versus $\ell^2$' + '\\n' + 'relative error.')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()


# =====================================================================
# ---- Gráfico 6b: Tempo de Avaliação vs Erro Relativo ----
# =====================================================================
if len(fem_err) > 0:
    if len(fem_eval_time) > 0:
        ax2.loglog(fem_err, fem_eval_time, linestyle='-', color='steelblue', label='FEM evaluation time', linewidth=2)
    ax2.loglog(fem_err, fem_time, linestyle='-', color='#9467bd', label='FEM solving time', linewidth=2) # Roxo

# Plotar dados
for width, metrics in sorted(pinn_data.items()):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    eval_time_sorted = np.array(metrics['eval_time'])[sort_indices]
    
    c = color_map.get(width, 'gray')
    ax2.loglog(err_sorted, eval_time_sorted, linestyle='-', color=c, alpha=0.9, 
               label=f'{width} nodes PINNs')
    ax2.loglog(err_sorted, eval_time_sorted, marker='o', linestyle='none', color='darkorange', alpha=1.0, zorder=3)

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
ax2.set_title('(b) Plot of time to interpolate FEM and evaluate PINN in sec' + '\\n' + 'versus relative error. For comparison, the time to solve FEM is' + '\\n' + 'also plotted.')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.suptitle(r'Figure 6: Plot for 3D Poisson equation of time in sec versus $\ell^2$ relative error.', fontsize=14, y=0.0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

plt.savefig("POISSON-3D-FEM-PINNs.png", dpi=150)
plt.show()