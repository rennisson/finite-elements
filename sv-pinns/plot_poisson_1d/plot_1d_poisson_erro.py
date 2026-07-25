import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os

PRINT_PAPER = True # Flag para printar ou nao os resultados do PAPER

# 1. Carregar dados do FEM da pasta correta
fem_file = "FEM/fem_results_1d.json" 
if os.path.exists(fem_file):
    with open(fem_file, "r") as f:
        fem_data = json.load(f)
    fem_err = np.array(fem_data["rel_error"])
    fem_time = np.array(fem_data["solve_time"])
else:
    print(f"Arquivo '{fem_file}' não encontrado. O plot do FEM será omitido.")
    fem_err, fem_time = [], []

# Dicionários separados para armazenar os diferentes dados
pinn_data = {}
svpinn_data = {}
pinn_data_paper = {}

# Função auxiliar para ler JSONs de Redes Neurais (PINNs e SVPINNs)
def load_nn_data(file_pattern, data_dict):
    files = glob.glob(file_pattern)
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                
            err = float(data['error_relativo_medio'])
            
            # Tempo de treino (time_training + time_training_lbfgs)
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
            
            if layers not in data_dict:
                data_dict[layers] = {'err': [], 'train_time': [], 'eval_time': []}
                
            data_dict[layers]['err'].append(err)
            data_dict[layers]['train_time'].append(train_t)
            data_dict[layers]['eval_time'].append(eval_t)
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")

# 2. Carregar e organizar dados das PINNs e SVPINNs
load_nn_data("PINN/dados_pinn_1d_*.json", pinn_data)
load_nn_data("SVPINN/dados_svpinn_1d_*.json", svpinn_data)

# 3. Carregar dados estruturados do arquivo JSON extraído do Paper Original
eval_file = "PINN/PINNs_1D_evaluation.json"
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
        layers = len(arch)
        
        err = l2_rel_dict[key]
        t_total = times_total_dict[key]
        t_eval = times_eval_dict[key]
        
        if layers not in pinn_data_paper:
            pinn_data_paper[layers] = {'err': [], 'train_time': [], 'eval_time': []}
            
        pinn_data_paper[layers]['err'].append(err)
        pinn_data_paper[layers]['train_time'].append(t_total)
        pinn_data_paper[layers]['eval_time'].append(t_eval)

# 4. Criar a Figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Paleta de cores baseada em índices
colors = ["#067E06", "#ff0000", "#ff7700"]

def get_color(layers_count):
    return colors[layers_count % len(colors)]

# =====================================================================
# ---- Gráfico 2a: Tempo de Treinamento/Solução vs Erro Relativo ----
# =====================================================================
if len(fem_err) > 0:
    ax1.loglog(fem_err, fem_time, linestyle='-', color='steelblue', label='FEM', linewidth=2)

# Plotar dados das PINNs (Linha tracejada + Marcador 'o')
for layers, metrics in sorted(pinn_data.items()):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    time_sorted = np.array(metrics['train_time'])[sort_indices]
    
    c = get_color(layers)
    ax1.loglog(err_sorted, time_sorted, linestyle='--', marker='o', color=c, alpha=0.9, 
               label=f'{layers}-layer PINNs')

# Plotar dados das SVPINNs (Linha traço-ponto + Marcador '^')
for layers, metrics in sorted(svpinn_data.items()):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    time_sorted = np.array(metrics['train_time'])[sort_indices]
    
    c = get_color(layers)
    ax1.loglog(err_sorted, time_sorted, linestyle=':', marker='s', color=c, alpha=0.9, 
               label=f'{layers}-layer SVPINNs')

if PRINT_PAPER:
    # Plotar dados do Paper (Linha sólida + Marcador 'D')
    for layers, metrics in sorted(pinn_data_paper.items()):
        sort_indices = np.argsort(metrics['err'])
        err_sorted = np.array(metrics['err'])[sort_indices]
        time_sorted = np.array(metrics['train_time'])[sort_indices]
        
        c = get_color(layers)
        ax1.loglog(err_sorted, time_sorted, linestyle='-', marker='^', markeredgecolor='black', 
                   color=c, linewidth=1.5, zorder=5, label=f'{layers}-layer PINNs (Paper)')

ax1.set_xlabel(r'Relative $\ell^2$ Error')
ax1.set_ylabel('Total time to solve in sec')
ax1.set_title(r'(a) Plot of time to solve FEM and train models versus $\ell^2$ relative error.')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend(fontsize=9) # Ajustado o tamanho da fonte da legenda para caber melhor

# =====================================================================
# ---- Gráfico 2b: Tempo de Avaliação vs Erro Relativo ----
# =====================================================================
if len(fem_err) > 0:
    ax2.loglog(fem_err, fem_time, linestyle='-', color='rosybrown', label='FEM solving time', linewidth=2)

# Plotar dados das PINNs (Linha tracejada + Marcador 'o')
for layers, metrics in sorted(pinn_data.items()):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    eval_time_sorted = np.array(metrics['eval_time'])[sort_indices]
    
    c = get_color(layers)
    ax2.loglog(err_sorted, eval_time_sorted, linestyle='--', marker='o', color=c, alpha=0.9, 
               label=f'{layers}-layer PINNs')

# Plotar dados das SVPINNs (Linha traço-ponto + Marcador '^')
for layers, metrics in sorted(svpinn_data.items()):
    sort_indices = np.argsort(metrics['err'])
    err_sorted = np.array(metrics['err'])[sort_indices]
    eval_time_sorted = np.array(metrics['eval_time'])[sort_indices]
    
    c = get_color(layers)
    ax2.loglog(err_sorted, eval_time_sorted, linestyle=':', marker='s', color=c, alpha=0.9, 
               label=f'{layers}-layer SVPINNs')

if PRINT_PAPER:
    # Plotar dados do Paper (Linha sólida + Marcador 'D')
    for layers, metrics in sorted(pinn_data_paper.items()):
        sort_indices = np.argsort(metrics['err'])
        err_sorted = np.array(metrics['err'])[sort_indices]
        eval_time_sorted = np.array(metrics['eval_time'])[sort_indices]
        
        c = get_color(layers)
        ax2.loglog(err_sorted, eval_time_sorted, linestyle='-', marker='^', markeredgecolor='black', 
                   color=c, linewidth=1.5, zorder=5, label=f'{layers}-layer PINNs (Paper)')

ax2.set_xlabel(r'Relative $\ell^2$ Error')
ax2.set_ylabel('Time in sec')
ax2.set_title(r'(b) Plot of time to interpolate FEM and evaluate models versus relative error.')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend(fontsize=9)

plt.tight_layout()

plt.savefig('POISSON-1D-FEM-PINNs-SVPINNs.png', dpi=150)
plt.show()