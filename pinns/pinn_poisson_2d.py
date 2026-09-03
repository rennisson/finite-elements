# -*- coding: utf-8 -*-
import os

import jax
import jax.numpy as jnp
from jax import config
import optax
import jaxopt

from scipy.stats import qmc
import numpy as np
import time
import json
from pathlib import Path

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ==========================================
# 1. CARREGAMENTO DO GROUND TRUTH E PDE
# ==========================================
gt_file = "gt_poisson_2d.json"
if not os.path.exists(gt_file):
    raise FileNotFoundError(f"Arquivo '{gt_file}' não encontrado. Gere o Ground Truth primeiro.")

with open(gt_file, 'r') as f:
    gt_data = json.load(f)

X_mesh_gt = jnp.array(gt_data["X"])
Y_mesh_gt = jnp.array(gt_data["Y"])
U_true_gt = jnp.array(gt_data["U_true"])

# Pontos e valores de referência achatados, usados nas avaliações de trajetória
xy_points_gt = jnp.stack([X_mesh_gt.ravel(), Y_mesh_gt.ravel()], axis=-1)
U_true_flat = U_true_gt.ravel()

print(f"Ground truth carregado. Shape da malha de avaliação: {X_mesh_gt.shape}")

def pde_rhs(x, y):
    return 2 * (x**4 * (3*y - 2) + x**3 * (4 - 6*y) + x**2 * (6*y**3 - 12*y**2 + 9*y - 2)
                - 6*x*(y - 1)**2 * y + (y - 1)**2 * y)

# ==========================================
# 2. DEFINIÇÕES GERAIS DA REDE E OTIMIZADOR
# ==========================================
activation_function = jax.nn.tanh
lr = 1e-3
epochs_adam = 15000
lbfgs_maxiter = 20000
eval_freq = 10
num_runs = 10
main_key = jax.random.PRNGKey(42)

N_f = 2000
N_g = 250

optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-08)

@jax.jit
def forward(xy, params):
    x = xy
    *hidden, output = params
    for layer in hidden:
        x = activation_function(x @ layer['W'] + layer['B'])
    return x @ output['W'] + output['B']

def u_net(x, y, params):
    xy = jnp.array([[x, y]])
    return forward(xy, params)[0, 0]

def laplacian_u(x, y, params):
    u_xx = jax.grad(jax.grad(lambda x_: u_net(x_, y, params), 0), 0)(x)
    u_yy = jax.grad(jax.grad(lambda y_: u_net(x, y_, params), 0), 0)(y)
    return u_xx + u_yy

def du_dx(x, y, params): return jax.grad(lambda x_: u_net(x_, y, params), 0)(x)
def du_dy(x, y, params): return jax.grad(lambda y_: u_net(x, y_, params), 0)(y)

def loss_function_interior(params, xy_interior):
    def residual(xy):
        x, y = xy[0], xy[1]
        return (laplacian_u(x, y, params) - pde_rhs(x, y))**2
    return jnp.mean(jax.vmap(residual)(xy_interior))

def loss_bc_x0(params, xy): return jnp.mean(jax.vmap(lambda pt: du_dx(pt[0], pt[1], params)**2)(xy))
def loss_bc_x1(params, xy): return jnp.mean(jax.vmap(lambda pt: du_dx(pt[0], pt[1], params)**2)(xy))
def loss_bc_y0(params, xy): return jnp.mean(jax.vmap(lambda pt: u_net(pt[0], pt[1], params)**2)(xy))
def loss_bc_y1(params, xy): return jnp.mean(jax.vmap(lambda pt: du_dy(pt[0], pt[1], params)**2)(xy))

@jax.jit
def loss_function(params, xy_interior, xy_x0, xy_x1, xy_y0, xy_y1):
    return (loss_function_interior(params, xy_interior) +
            loss_bc_x0(params, xy_x0) + loss_bc_x1(params, xy_x1) +
            loss_bc_y0(params, xy_y0) + loss_bc_y1(params, xy_y1))

grad_loss = jax.jit(jax.grad(loss_function, 0))

@jax.jit
def update(opt_state, params, xy_interior, xy_x0, xy_x1, xy_y0, xy_y1):
    grads = grad_loss(params, xy_interior, xy_x0, xy_x1, xy_y0, xy_y1)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

def generate_lhs_points(n_f, n_g):
    sampler_f = qmc.LatinHypercube(d=2)
    xy_interior = jnp.array(sampler_f.random(n=n_f), dtype=jnp.float64)

    sampler_g = qmc.LatinHypercube(d=1)
    y_x0 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_x0 = jnp.column_stack([jnp.zeros(n_g), y_x0])
    y_x1 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_x1 = jnp.column_stack([jnp.ones(n_g), y_x1])
    x_y0 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_y0 = jnp.column_stack([x_y0, jnp.zeros(n_g)])
    x_y1 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_y1 = jnp.column_stack([x_y1, jnp.ones(n_g)])
    
    return xy_interior, xy_x0, xy_x1, xy_y0, xy_y1

# ==========================================
# 3. LOOP PRINCIPAL DE TREINAMENTO 
# ==========================================
architectures = [
    [20, 20, 1], [60, 60, 1],
    [20, 20, 20, 1], [60, 60, 60, 1],
    [20, 20, 20, 20, 1], [60, 60, 60, 60, 1],
    [20, 20, 20, 20, 20, 1], [60, 60, 60, 60, 60, 1],
    [120, 120, 120, 120, 120, 1]
]

output_dir = Path("pinn_poisson_2d")
output_dir.mkdir(exist_ok=True)

for arch in architectures:
    width = [2] + arch # Adiciona a dimensão de entrada (2)
    arch_str = "_".join(map(str, width))
    
    print("\n" + "="*50)
    print(f"INICIANDO: ARQUITETURA {width} | {num_runs} EXECUÇÕES")
    print("="*50)

    acc_train_adam = 0.0
    acc_train_lbfgs = 0.0
    acc_eval_time = 0.0
    l2_errors = [] 

    all_loss_trajectories = []           # (num_runs, lbfgs_maxiter // eval_freq)
    all_l2_error_trajectories = []       # (num_runs, lbfgs_maxiter // eval_freq)
    all_adam_loss_trajectories = []      # (num_runs, epochs_adam // eval_freq)
    all_adam_l2_error_trajectories = []  # (num_runs, epochs_adam // eval_freq)
    
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
    def objective_lbfgs(p_flat, xy_interior, xy_x0, xy_x1, xy_y0, xy_y1):
        return loss_function(unflatten_fn(p_flat), xy_interior, xy_x0, xy_x1, xy_y0, xy_y1)
    
    lbfgs = jaxopt.LBFGS(fun=objective_lbfgs, maxiter=lbfgs_maxiter, history_size=200, tol=1e-12)

    # Captura o erro L2 a cada eval_freq passos de L-BFGS (mesmo padrão do 1D)
    @jax.jit
    def lbfgs_segment_with_trajectory(p_flat, xy_interior, xy_x0, xy_x1, xy_y0, xy_y1):
        init_state = lbfgs.init_state(p_flat, xy_interior=xy_interior, xy_x0=xy_x0,
                                       xy_x1=xy_x1, xy_y0=xy_y0, xy_y1=xy_y1)

        num_blocks = lbfgs_maxiter // eval_freq

        def block_fn(carry, _):
            p, state = carry

            def inner_step(i, val):
                p_in, state_in = val
                p_out, state_out = lbfgs.update(p_in, state_in, xy_interior=xy_interior,
                                                 xy_x0=xy_x0, xy_x1=xy_x1, xy_y0=xy_y0, xy_y1=xy_y1)
                return p_out, state_out

            p_next, state_next = jax.lax.fori_loop(0, eval_freq, inner_step, (p, state))

            params_ = unflatten_fn(p_next)
            u_nn_step = forward(xy_points_gt, params_).flatten()
            l2_err = jnp.linalg.norm(u_nn_step - U_true_flat) / jnp.linalg.norm(U_true_flat)

            return (p_next, state_next), (state_next.value, l2_err)

        (p_final, state_final), (loss_traj, err_traj) = jax.lax.scan(
            block_fn, (p_flat, init_state), xs=None, length=num_blocks)

        return p_final, state_final, loss_traj, err_traj

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

        adam_loss_traj_run = []
        adam_l2_traj_run = []

        # 2. Treinamento Adam
        t_start_adam = time.perf_counter()
        for e in range(epochs_adam):
            xy_int, x0, x1, y0, y1 = generate_lhs_points(N_f, N_g)
            opt_state, params = update(opt_state, params, xy_int, x0, x1, y0, y1)

            if (e + 1) % eval_freq == 0:
                u_nn_step = forward(xy_points_gt, params).flatten()
                l2_err_step = jnp.linalg.norm(u_nn_step - U_true_flat) / jnp.linalg.norm(U_true_flat)
                loss_step = loss_function(params, xy_int, x0, x1, y0, y1)
                adam_loss_traj_run.append(float(loss_step))
                adam_l2_traj_run.append(float(l2_err_step))
        
        t_adam = time.perf_counter() - t_start_adam
        acc_train_adam += t_adam
        
        # Avaliar loss final do adam para logs
        xy_int, x0, x1, y0, y1 = generate_lhs_points(N_f, N_g)
        loss_adam = loss_function(params, xy_int, x0, x1, y0, y1).block_until_ready()
        print(f"Adam concluído em {t_adam:.2f}s | Loss: {loss_adam:.8e}")

        all_adam_loss_trajectories.append(np.asarray(adam_loss_traj_run))
        all_adam_l2_error_trajectories.append(np.asarray(adam_l2_traj_run))

        # 3. Refinamento L-BFGS
        params_flat, _ = jax.flatten_util.ravel_pytree(params)
        
        xy_int_lbfgs, x0_lbfgs, x1_lbfgs, y0_lbfgs, y1_lbfgs = generate_lhs_points(N_f, N_g)
        
        t_start_lbfgs = time.perf_counter()
        params_flat, state, loss_traj_run, err_traj_run = lbfgs_segment_with_trajectory(
            params_flat, xy_int_lbfgs, x0_lbfgs, x1_lbfgs, y0_lbfgs, y1_lbfgs
        )
        t_lbfgs = time.perf_counter() - t_start_lbfgs
        acc_train_lbfgs += t_lbfgs
        
        params = unflatten_fn(params_flat).copy() 
        loss_lbfgs = loss_function(params, xy_int_lbfgs, x0_lbfgs, x1_lbfgs, y0_lbfgs, y1_lbfgs).block_until_ready()
        print(f"L-BFGS concluído em {t_lbfgs:.2f}s | Loss: {loss_lbfgs:.8e}")

        loss_trajectory_run = np.asarray(loss_traj_run)
        l2_error_trajectory_run = np.asarray(err_traj_run)
        all_loss_trajectories.append(loss_trajectory_run)
        all_l2_error_trajectories.append(l2_error_trajectory_run)

        # 4. Avaliação e Erro L2
        _ = forward(xy_points_gt, params).block_until_ready()
        
        t_start_eval = time.perf_counter()
        u_nn_flat = forward(xy_points_gt, params)
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

    # 5. Salvar Dados (.json) - mesma estrutura de 3 arquivos usada no 1D
    results = {
        'architecture': width,
        'num_hidden_layers': len(width) - 1,
        'epochs_adam': epochs_adam,
        'lbfgs_maxiter': lbfgs_maxiter,
        'num_runs_avg': num_runs,
        'error_relativo_medio': avg_rel_l2, 
        'error_relativo_mediana': median_rel_l2, 
        'time_training': avg_train_adam,
        'time_training_lbfgs': avg_train_total,
        'time_evaluation': avg_eval_time,
        'num_params': int(len(params_final_flat))
    }

    nome_arquivo = output_dir / f'dados_pinn_2d_{arch_str}.json'
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    points = {
        'x': X_mesh_gt.ravel().tolist(),
        'y': Y_mesh_gt.ravel().tolist(),
        'u_nn': U_nn_final.ravel().tolist(),
        'network_weights': params_final_flat.tolist()
    }

    nome_arquivo = output_dir / f'pontos_pinn_2d_{arch_str}.json'
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(points, f, indent=4)

    all_loss_trajectories = np.stack(all_loss_trajectories, axis=0)
    all_l2_error_trajectories = np.stack(all_l2_error_trajectories, axis=0)
    all_adam_loss_trajectories = np.stack(all_adam_loss_trajectories, axis=0)
    all_adam_l2_error_trajectories = np.stack(all_adam_l2_error_trajectories, axis=0)

    training_curve = {
        'architecture': width,
        'method': 'PINN',
        'epochs_adam': epochs_adam,
        'lbfgs_maxiter': lbfgs_maxiter,
        'num_runs': num_runs,
        'steps_adam': list(range(eval_freq, epochs_adam + 1, eval_freq)),
        'l2_relative_error_per_run_adam': all_adam_l2_error_trajectories.tolist(),
        'loss_per_run_adam': all_adam_loss_trajectories.tolist(),
        'l2_relative_error_mean_adam': all_adam_l2_error_trajectories.mean(axis=0).tolist(),
        'l2_relative_error_std_adam': all_adam_l2_error_trajectories.std(axis=0).tolist(),
        'steps': list(range(eval_freq, lbfgs_maxiter + 1, eval_freq)),
        'l2_relative_error_per_run': all_l2_error_trajectories.tolist(),
        'loss_per_run': all_loss_trajectories.tolist(),
        'l2_relative_error_mean': all_l2_error_trajectories.mean(axis=0).tolist(),
        'l2_relative_error_std': all_l2_error_trajectories.std(axis=0).tolist(),
    }

    nome_arquivo = output_dir / f'curva_treino_pinn_2d_{arch_str}.json'
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(training_curve, f, indent=4)

    print(f"Dados salvos como: {nome_arquivo}\n")