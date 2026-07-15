import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os

PRINT_PAPER = False # Flag para printar ou nao os resultados do PAPER

# 1. Carregar dados do FEM
fem_file = "npz/fem_results_1d.json" 
if os.path.exists(fem_file):
    with open(fem_file, "r") as f:
        fem_data = json.load(f)
    fem_err = np.array(fem_data["rel_error"])
    fem_time = np.array(fem_data["solve_time"])
else:
    print(f"Arquivo '{fem_file}' não encontrado. O plot do FEM será omitido.")
    fem_err, fem_time = [], []

# Dicionários separados para não misturar os dados
pinn_data = {}
pinn_data_paper = {}

# 2. Carregar e organizar dados das PINNs
# Alteramos para procurar todos os json na pasta
pinn_files = glob.glob("npz/results_poisson_1d/dados_pinn_1d_*.json")

for file in pinn_files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            
        err = float(data['error_relativo_medio'])
        
        # Como visto no JSON enviado, tempo de treino muitas vezes 
        # está separado (time_training + time_training_lbfgs) ou 
        # só time_training se não usou lbfgs
        train_t = float(data.get('time_training', 0.0))
        if 'time_training_lbfgs' in data:
            train_t += float(data['time_training_lbfgs'])
            
        eval_t = float(data['time_evaluation'])
        
        # Extrair número de camadas
        if 'num_hidden_layers' in data:
            layers = int(data['num_hidden_layers'])
        elif 'architecture' in data:
            arch = data['architecture']
            layers = len(arch) - 2 # Input e Output
        else:
            print(f"Não foi possível determinar a arquitetura para {file}")
            continue
        
        if layers not in pinn_data:
            pinn_data[layers] = {'err': [], 'train_time': [], 'eval_time': []}
            
        pinn_data[layers]['err'].append(err)
        pinn_data[layers]['train_time'].append(train_t)
        pinn_data[layers]['eval_time'].append(eval_t)
    except Exception as e:
        print(f"Erro ao ler {file}: {e}")

# 3. Carregar dados estruturados do arquivo JSON extraído do Paper Original
eval_file = "PINNs_1D_evaluation.json"
if os.path.exists(eval_file):
    with open(eval_file, "r") as f:
        eval_data = json.load(f)
    
    arch_dict = eval_data.get("arch", {})
    l2_rel_dict = eval_data.get("l2_rel", {})
    times_total_dict = eval_data.get("times_total", {})
    times_eval_dict = eval_data.get("times_eval", {})
    
    # Ordenar as chaves numericamente
    sorted_keys = sorted(l2_rel_dict.keys(), key=lambda x: int(x))
    
    for key in sorted_keys:
        arch = arch_dict[key]
        
        # Assumindo a notação [w, w, 1] ou [w, 1] no JSON, onde o input não foi escrito explicitamente.
        # Logo, o número de camadas ocultas é o tamanho da lista menos 1 (a saída)
        layers = len(arch)
        
        err = l2_rel_dict[key]
        t_total = times_total_dict[key]
        t_eval = times_eval_dict[key]
        
        if layers not in pinn_data_paper:
            pinn_data_paper[layers] = {'err': [], 'train_time': [], 'eval_time': []}
            
        pinn_data_paper[layers]['err'].append(err)
        pinn_data_paper[layers]['train_time'].append(t_total)
        pinn_data_paper[layers]['eval_time'].append(t_eval)
else:
    print(f"Arquivo '{eval_file}' não encontrado.")

# 4. Criar a Figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Paleta de cores baseada em índices (para 1D geralmente separamos por nº de camadas)
colors = ['#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#e377c2', '#8c564b', '#1f77b4']

# Função auxiliar para garantir mesma cor por quantidade de camadas
def get_color(layers_count):
    # Usa a quantidade de camadas como índice, garantindo a mesma cor em ambos datasets
    return colors[layers_count % len(colors)]

# =====================================================================
# ---- Gráfico 2a: Tempo de Treinamento/Solução vs Erro Relativo ----
# =====================================================================
if len(fem_err) > 0:
    ax1.loglog(fem_err, fem_time, linestyle='-', color='steelblue', label='FEM', linewidth=2)

# Plotar dados
for layers, metrics in sorted(pinn_data.items()):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    time_sorted = np.array(metrics['train_time'])[sort_indices]
    
    c = get_color(layers)
    ax1.loglog(err_sorted, time_sorted, linestyle='--', color=c, alpha=0.9, 
               label=f'{layers}-layer PINNs')
    ax1.loglog(err_sorted, time_sorted, marker='o', linestyle='none', color=c, alpha=0.9)

if PRINT_PAPER:
    # Plotar NOVOS dados do JSON em destaque
    for layers, metrics in sorted(pinn_data_paper.items()):
        sort_indices = np.argsort(metrics['err'])
        err_sorted = np.array(metrics['err'])[sort_indices]
        time_sorted = np.array(metrics['train_time'])[sort_indices]
        
        c = get_color(layers)
        ax1.loglog(err_sorted, time_sorted, linestyle='-', color=c, linewidth=1.5,
                label=f'{layers}-layer PINNs (Paper)')
        ax1.loglog(err_sorted, time_sorted, marker='D', markeredgecolor='black', 
                linestyle='none', color=c, zorder=5)

ax1.set_xlabel(r'Relative $\ell^2$ Error')
ax1.set_ylabel('Total time to solve in sec')
ax1.set_title(r'(a) Plot of time to solve FEM and train PINN versus $\ell^2$ relative error.')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# =====================================================================
# ---- Gráfico 2b: Tempo de Avaliação vs Erro Relativo ----
# =====================================================================
if len(fem_err) > 0:
    ax2.loglog(fem_err, fem_time, linestyle='-', color='rosybrown', label='FEM solving time', linewidth=2)

# Plotar dados
for layers, metrics in sorted(pinn_data.items()):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    eval_time_sorted = np.array(metrics['eval_time'])[sort_indices]
    
    c = get_color(layers)
    ax2.loglog(err_sorted, eval_time_sorted, linestyle='--', color=c, alpha=0.9, 
               label=f'{layers}-layer PINNs')
    ax2.loglog(err_sorted, eval_time_sorted, marker='o', linestyle='none', color=c, alpha=0.9)

if PRINT_PAPER:
    # Plotar NOVOS dados do JSON em destaque
    for layers, metrics in sorted(pinn_data_paper.items()):
        sort_indices = np.argsort(metrics['err'])
        err_sorted = np.array(metrics['err'])[sort_indices]
        eval_time_sorted = np.array(metrics['eval_time'])[sort_indices]
        
        c = get_color(layers)
        ax2.loglog(err_sorted, eval_time_sorted, linestyle='-', color=c, linewidth=1.5,
                label=f'{layers}-layer PINNs (Paper)')
        ax2.loglog(err_sorted, eval_time_sorted, marker='D', markeredgecolor='black', 
                linestyle='none', color=c, zorder=5)

ax2.set_xlabel(r'Relative $\ell^2$ Error')
ax2.set_ylabel('Time in sec')
ax2.set_title(r'(b) Plot of time to interpolate FEM and evaluate PINN in sec versus relative error.')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

# plt.suptitle(r'Figure 2: Plot for 1D Poisson equation of time in sec versus $\ell^2$ relative error.', fontsize=14)
plt.tight_layout()

plt.savefig('POISSON-1D-FEM-PINNs.png', dpi=150)
plt.show()