# -*- coding: utf-8 -*-
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax import config
import optax
import jaxopt

from scipy.stats import qmc
import numpy as np
import time
import json

# Forçar uso de float64 para maior precisão, mantendo a convenção validada do 2D
config.update("jax_enable_x64", True)

# ==========================================
# 1. CARREGAMENTO DO GROUND TRUTH E PDE
# ==========================================
# Carrega a malha exata de 150x150x150 conforme descrito no artigo
gt_file = "gt_poisson_3d.json"
if not os.path.exists(gt_file):
    raise FileNotFoundError(f"Arquivo '{gt_file}' não encontrado. Gere o Ground Truth primeiro.")

print("Carregando Ground Truth (Isso pode levar alguns segundos devido ao tamanho do grid 150^3)...")
with open(gt_file, 'r') as f:
    gt_data = json.load(f)

X_mesh_gt = jnp.array(gt_data["X"])
Y_mesh_gt = jnp.array(gt_data["Y"])
Z_mesh_gt = jnp.array(gt_data["Z"])
U_true_gt = jnp.array(gt_data["U_true"])

print(f"Ground truth carregado. Shape da malha de avaliação: {X_mesh_gt.shape}")

# Lado direito (RHS) da PDE conforme a Equação de Poisson 3D do paper
def pde_rhs(x, y, z):
    return -3 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)

# ==========================================
# 2. DEFINIÇÕES GERAIS DA REDE E OTIMIZADOR
# ==========================================
activation_function = jax.nn.tanh
lr = 1e-3
epochs_adam = 20000
num_runs = 10
main_key = jax.random.PRNGKey(42)

# Valores amostrais definidos explicitamente na seção 4.3 do paper
N_f = 1000  # Collocation points no domínio
N_g = 100   # Collocation points no contorno (para CADA face/fronteira)

optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-08)

@jax.jit
def forward(xyz, params):
    x = xyz
    *hidden, output = params
    for layer in hidden:
        x = activation_function(x @ layer['W'] + layer['B'])
    return x @ output['W'] + output['B']

def u_net(x, y, z, params):
    xyz = jnp.array([[x, y, z]])
    return forward(xyz, params)[0, 0]

def laplacian_u(x, y, z, params):
    u_xx = jax.grad(jax.grad(lambda x_: u_net(x_, y, z, params), 0), 0)(x)
    u_yy = jax.grad(jax.grad(lambda y_: u_net(x, y_, z, params), 0), 0)(y)
    u_zz = jax.grad(jax.grad(lambda z_: u_net(x, y, z_, params), 0), 0)(z)
    return u_xx + u_yy + u_zz

def loss_function_interior(params, xyz_interior):
    def residual(xyz):
        x, y, z = xyz[0], xyz[1], xyz[2]
        return (laplacian_u(x, y, z, params) - pde_rhs(x, y, z))**2
    return jnp.mean(jax.vmap(residual)(xyz_interior))

# Condições de Contorno de Dirichlet (u = 0 em todas as 6 faces da fronteira)
def loss_bc_face(params, xyz_face):
    return jnp.mean(jax.vmap(lambda pt: u_net(pt[0], pt[1], pt[2], params)**2)(xyz_face))

@jax.jit
def loss_function(params, xyz_interior, xyz_x0, xyz_x1, xyz_y0, xyz_y1, xyz_z0, xyz_z1):
    return (loss_function_interior(params, xyz_interior) +
            loss_bc_face(params, xyz_x0) + loss_bc_face(params, xyz_x1) +
            loss_bc_face(params, xyz_y0) + loss_bc_face(params, xyz_y1) +
            loss_bc_face(params, xyz_z0) + loss_bc_face(params, xyz_z1))

grad_loss = jax.jit(jax.grad(loss_function, 0))

@jax.jit
def update(opt_state, params, xyz_interior, xyz_x0, xyz_x1, xyz_y0, xyz_y1, xyz_z0, xyz_z1):
    grads = grad_loss(params, xyz_interior, xyz_x0, xyz_x1, xyz_y0, xyz_y1, xyz_z0, xyz_z1)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

def generate_lhs_points(n_f, n_g):
    """Gera pontos LHS para o interior 3D e para as 6 faces do cubo unitário."""
    # Pontos interiores (3D)
    sampler_f = qmc.LatinHypercube(d=3)
    xyz_interior = jnp.array(sampler_f.random(n=n_f), dtype=jnp.float64)

    # Pontos de fronteira (Planos 2D em cada face)
    sampler_g = qmc.LatinHypercube(d=2)
    
    # Planos X=0 e X=1 (y, z variam)
    pts_x = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64)
    xyz_x0 = jnp.column_stack([jnp.zeros(n_g), pts_x[:, 0], pts_x[:, 1]])
    xyz_x1 = jnp.column_stack([jnp.ones(n_g), pts_x[:, 0], pts_x[:, 1]])

    # Planos Y=0 e Y=1 (x, z variam)
    pts_y = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64)
    xyz_y0 = jnp.column_stack([pts_y[:, 0], jnp.zeros(n_g), pts_y[:, 1]])
    xyz_y1 = jnp.column_stack([pts_y[:, 0], jnp.ones(n_g), pts_y[:, 1]])

    # Planos Z=0 e Z=1 (x, y variam)
    pts_z = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64)
    xyz_z0 = jnp.column_stack([pts_z[:, 0], pts_z[:, 1], jnp.zeros(n_g)])
    xyz_z1 = jnp.column_stack([pts_z[:, 0], pts_z[:, 1], jnp.ones(n_g)])
    
    return xyz_interior, xyz_x0, xyz_x1, xyz_y0, xyz_y1, xyz_z0, xyz_z1

# ==========================================
# 3. LOOP PRINCIPAL DE TREINAMENTO 
# ==========================================
# Arquiteturas idênticas às testadas no arquivo 3D_Poisson e citadas no artigo
architectures = [
    [20, 20, 1], 
    [60, 60, 1], 
    [20, 20, 20, 1], 
    [60, 60, 60, 1],
    [20, 20, 20, 20, 1], 
    [60, 60, 60, 60, 1], 
    [20, 20, 20, 20, 20, 1], 
    [60, 60, 60, 60, 60, 1]
]

for arch in architectures:
    width = [3] + arch # Adiciona a dimensão de entrada (3) para X,Y,Z
    arch_str = "_".join(map(str, width))
    
    print("\n" + "="*10)
    print(f"INICIANDO: ARQUITETURA {width} | {num_runs} EXECUÇÕES")
    print("="*10)

    acc_train_adam = 0.0
    acc_train_lbfgs = 0.0
    acc_eval_time = 0.0
    l2_errors = [] 
    
    U_nn_final = None
    params_final_flat = None

    # Configuração do descompactador (unflatten_fn) para a arquitetura atual
    initializer = jax.nn.initializers.glorot_normal()
    dummy_key = jax.random.split(main_key, len(width) - 1)
    dummy_params = []
    for k, lin, lout in zip(dummy_key, width[:-1], width[1:]):
        dummy_params.append({'W': jnp.zeros((lin, lout), dtype=jnp.float64), 
                             'B': jnp.zeros((1, lout), dtype=jnp.float64)})
        
    _, unflatten_fn = jax.flatten_util.ravel_pytree(dummy_params)

    @jax.jit
    def objective_lbfgs(p_flat, xyz_int, x0, x1, y0, y1, z0, z1):
        return loss_function(unflatten_fn(p_flat), xyz_int, x0, x1, y0, y1, z0, z1)
    
    lbfgs = jaxopt.LBFGS(fun=objective_lbfgs, maxiter=50000, history_size=50, tol=1e-12)

    # --- LOOP DE EXECUÇÕES (RUNS) ---
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        # 1. Inicializar parâmetros com semente única
        run_key = jax.random.fold_in(main_key, run)
        layer_keys = jax.random.split(run_key, len(width) - 1)
        params = []
        for k, lin, lout in zip(layer_keys, width[:-1], width[1:]):
            W = initializer(k, (lin, lout), jnp.float64)
            B = initializer(k, (1, lout), dtype=jnp.float64)
            params.append({'W': W, 'B': B})

        opt_state = optimizer.init(params)

        # 2. Treinamento Adam
        t_start_adam = time.perf_counter()
        for e in range(epochs_adam):
            xyz_int, x0, x1, y0, y1, z0, z1 = generate_lhs_points(N_f, N_g)
            opt_state, params = update(opt_state, params, xyz_int, x0, x1, y0, y1, z0, z1)
        
        t_adam = time.perf_counter() - t_start_adam
        acc_train_adam += t_adam
        
        # Avaliar loss final do adam para logs
        xyz_int, x0, x1, y0, y1, z0, z1 = generate_lhs_points(N_f, N_g)
        loss_adam = loss_function(params, xyz_int, x0, x1, y0, y1, z0, z1).block_until_ready()
        print(f"Adam concluído em {t_adam:.2f}s | Loss: {loss_adam:.8e}")

        # 3. Refinamento L-BFGS
        params_flat, _ = jax.flatten_util.ravel_pytree(params)
        
        xyz_int_lbfgs, x0_lbfgs, x1_lbfgs, y0_lbfgs, y1_lbfgs, z0_lbfgs, z1_lbfgs = generate_lhs_points(N_f, N_g)
        
        t_start_lbfgs = time.perf_counter()
        params_flat, state = lbfgs.run(params_flat, 
                                       xyz_int=xyz_int_lbfgs, 
                                       x0=x0_lbfgs, x1=x1_lbfgs, 
                                       y0=y0_lbfgs, y1=y1_lbfgs, 
                                       z0=z0_lbfgs, z1=z1_lbfgs)
        t_lbfgs = time.perf_counter() - t_start_lbfgs
        acc_train_lbfgs += t_lbfgs
        
        params = unflatten_fn(params_flat).copy() 
        loss_lbfgs = loss_function(params, xyz_int_lbfgs, x0_lbfgs, x1_lbfgs, y0_lbfgs, y1_lbfgs, z0_lbfgs, z1_lbfgs).block_until_ready()
        print(f"L-BFGS concluído em {t_lbfgs:.2f}s | Loss: {loss_lbfgs:.8e}")

        # 4. Avaliação e Erro L2 (Na malha 3D completa de 150x150x150)
        t_start_eval = time.perf_counter()
        xyz_points = jnp.stack([X_mesh_gt.ravel(), Y_mesh_gt.ravel(), Z_mesh_gt.ravel()], axis=-1)
        
        # Realizando inferência em batch completo (JAX lida bem com arrays flattenizados grandes,
        # mas caso enfrente limitações de GPU, um data loader para fatiamento pode ser injetado aqui)
        u_nn_flat = forward(xyz_points, params)
        u_nn = u_nn_flat.reshape(X_mesh_gt.shape)
        u_nn.block_until_ready()
        
        t_eval = time.perf_counter() - t_start_eval
        acc_eval_time += t_eval

        norma_erro = jnp.linalg.norm(u_nn - U_true_gt)
        norma_exata = jnp.linalg.norm(U_true_gt)
        rel_l2_error = float(norma_erro / norma_exata)
        l2_errors.append(rel_l2_error)
        print(f"Erro L2 Relativo (Run {run+1}): {rel_l2_error:.8e}") 
        
        if run == num_runs - 1:
            U_nn_final = np.asarray(u_nn)
            params_final_flat = np.asarray(params_flat)

    # --- CÁLCULO DAS MÉDIAS E EXPORTAÇÃO ---
    avg_train_adam = float(acc_train_adam / num_runs)
    avg_train_lbfgs = float(acc_train_lbfgs / num_runs)
    avg_train_total = float(avg_train_adam + avg_train_lbfgs) # Representa tempo acumulado
    avg_eval_time = float(acc_eval_time / num_runs)
    
    avg_rel_l2 = float(np.mean(l2_errors))
    median_rel_l2 = float(np.median(l2_errors))

    print("\n" + "-"*30)
    print(f"RESUMO DAS MÉDIAS ({num_runs} RUNS) - ARQUITETURA {width}")
    print(f"Erro L2 Relativo Médio: {avg_rel_l2:.8e}")
    print(f"Tempo Médio Treino Total: {avg_train_total:.3f}s")
    print("-" * 30)

    # 5. Salvar Dados (.json) idêntico à padronização validada
    results = {
        'architecture': width,
        'num_hidden_layers': len(width) - 1,
        'epochs_adam': epochs_adam,
        'num_runs_avg': num_runs,
        'error_relativo_medio': avg_rel_l2, 
        'error_relativo_mediana': median_rel_l2, 
        'time_training': avg_train_adam,
        'time_training_lbfgs': avg_train_total,
        'time_evaluation': avg_eval_time,
        'num_params': int(len(params_final_flat)),
        'y_nn': U_nn_final.tolist(),             
        'network_weights': params_final_flat.tolist() 
    }

    nome_arquivo = f'dados_pinn_3d_{arch_str}.json' 
    print("Exportando arquivo JSON (pode demorar alguns instantes por se tratar da malha 3D)...")
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"Dados salvos como: {nome_arquivo}\n")