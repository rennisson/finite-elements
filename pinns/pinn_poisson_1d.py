# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import jaxopt
import json
import numpy as np
import optax
import os
import time

from jax import config
from scipy.stats import qmc
from pathlib import Path

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

activation_function = jax.nn.tanh

def u(x):
    """Exact solution of the PDE"""
    return x * jnp.exp(-x**2)

def rhs(x):
    """Defines the right hand side of the equation"""
    return (4*x**3 - 6*x) * jnp.exp(-x**2)

# ==========================================
# 1. CONFIGURAÇÕES DO DOMÍNIO E GROUND TRUTH
# ==========================================
x_lower = 0
x_upper = 1

# Geração do Ground Truth com malha de pontos
gt_file = "gt_poisson_1d.json"
if not os.path.exists(gt_file):
    raise FileNotFoundError(f"Arquivo '{gt_file}' não encontrado. Por favor, verifique o diretório.")

with open(gt_file, "r") as f_in:
    gt_data = json.load(f_in)

# Assegurar que os arrays estão no formato de coluna, como esperado pela PINN
X_test = jnp.array(gt_data['X']).reshape(-1, 1)
Y_test = jnp.array(gt_data['U_true']).reshape(-1, 1)

print(f"Ground truth carregado com {X_test.shape[0]} pontos.")

# ==========================================
# 2. DADOS E CONFIGURAÇÕES DE TREINO
# ==========================================
n  = 256
key_x = jax.random.PRNGKey(136)

# Gera amostras usando Latin Hypercube Sampling no intervalo [0, 1)
sampler = qmc.LatinHypercube(d=1, seed=136)
sample = sampler.random(n=n)
X = jnp.array(sample, dtype=jnp.float32)

learning_rate = 1e-4
b1 = 0.9
b2 = 0.999
eps = 1e-08
eps_root = 0.0
epochs_adam = 15000

# Inicializa o otimizador Adam
optimizer = optax.adam(learning_rate, b1, b2, eps, eps_root)

# ==========================================
# 3. FUNÇÕES JAX
# ==========================================
@jax.jit
def forward(x, params):
    *hidden, output = params
    for layer in hidden:
        x = activation_function(x @ layer['W'] + layer['B'])
    return x @ output['W'] + output['B']

def laplacian(f, argnum=0):
    return jax.grad(jax.grad(f, argnums=argnum), argnums=argnum)

def pde(u_fn):
    return lambda x: laplacian(f=lambda x_: jnp.sum(u_fn(x_)), argnum=0)(x)

def MSE(pred, true):
    return jnp.mean((pred - true) ** 2)

def loss_interior(x, params):
    def u_net(x_val):
        x_v = x_val.reshape(1, 1)
        return forward(x_v, params)[0, 0]
    
    laplacian_u = jax.vmap(pde(u_net))(x.flatten())
    u_xx = laplacian_u.reshape(-1, 1)
    return MSE(u_xx, rhs(x))

@jax.jit
def loss_function(params, x):
    pde_loss = loss_interior(x, params)
    bc_0 = MSE(forward(jnp.array([[0.0]]), params), 0.0)
    bc_1 = MSE(forward(jnp.array([[1.0]]), params), jnp.exp(-1.0))
    return pde_loss + bc_0 + bc_1

grad_loss = jax.jit(jax.grad(loss_function, 0))

@jax.jit
def update(opt_state, params):
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=n)

    x = jnp.array(sample, dtype=jnp.float32)
    
    grads = grad_loss(params, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

# ==========================================
# 4. LOOP PRINCIPAL DE TREINAMENTO E MEDIÇÃO
# ==========================================
architectures = [
    [1, 1], [2, 1],
    [5, 1], [10, 1], [20, 1], [40, 1],
    [5, 5, 1], [10, 10, 1], [20, 20, 1], [40, 40, 1],
    [5, 5, 5, 1], [10, 10, 10, 1], [20, 20, 20, 1], [40, 40, 40, 1]
]

num_runs = 10 
main_key = jax.random.PRNGKey(42)

for arch in architectures:
    width = [1] + arch
    arch_str = "_".join(map(str, width))
    
    print("\n" + "="*20)
    print(f"INICIANDO: ARQUITETURA {width} | {num_runs} EXECUÇÕES")
    print("="*20)

    # Acumuladores de tempo e erro
    acc_train_adam = 0.0
    acc_train_lbfgs = 0.0
    acc_eval_time = 0.0
    l2_errors = [] 
    
    all_loss_trajectories = []       # (num_runs, lbfgs_maxiter // eval_freq)
    all_l2_error_trajectories = []   # (num_runs, lbfgs_maxiter // eval_freq)

    Y_nn_final = None
    params_final_flat = None

    # Compilar a função objetivo L-BFGS para a arquitetura atual fora do loop de medição
    initializer = jax.nn.initializers.glorot_normal()
    dummy_key = jax.random.split(main_key, len(width) - 1)
    dummy_params = []
    for k, lin, lout in zip(dummy_key, width[:-1], width[1:]):
        dummy_params.append({'W': jnp.zeros((lin, lout)), 'B': jnp.zeros((1, lout))})
        
    _, unflatten_fn = jax.flatten_util.ravel_pytree(dummy_params)

    @jax.jit
    def objective_lbfgs(p_flat, x_colloc):
        return loss_function(unflatten_fn(p_flat), x_colloc)
    
    lbfgs_maxiter = 50000
    lbfgs = jaxopt.LBFGS(fun=objective_lbfgs, maxiter=lbfgs_maxiter, history_size=200, tol=1e-12)

    # Captura o erro L2 a cada 10 passos.
    @jax.jit
    def lbfgs_segment_with_trajectory(p_flat, x_colloc):
        init_state = lbfgs.init_state(p_flat, x_colloc=x_colloc)
        
        eval_freq = 10
        # Reduz o tamanho do scan para iterar sobre blocos de passos
        num_blocks = lbfgs_maxiter // eval_freq

        def block_fn(carry, _):
            p, state = carry
            
            # Loop interno: executa 10 iteracoes apenas atualizando os pesos (sem inferencia)
            def inner_step(i, val):
                p_in, state_in = val
                p_out, state_out = lbfgs.update(p_in, state_in, x_colloc=x_colloc)
                return p_out, state_out
                
            p_next, state_next = jax.lax.fori_loop(0, eval_freq, inner_step, (p, state))
            
            # Avaliacao: ocorre apenas 1x ao final do bloco de 10 passos
            params_ = unflatten_fn(p_next)
            Y_nn_step = forward(X_test, params_)
            l2_err = jnp.linalg.norm(Y_nn_step - Y_test) / jnp.linalg.norm(Y_test)
            
            return (p_next, state_next), (state_next.value, l2_err)

        (p_final, state_final), (loss_traj, err_traj) = jax.lax.scan(
            block_fn, (p_flat, init_state), xs=None, length=num_blocks)
            
        return p_final, state_final, loss_traj, err_traj


    # --- LOOP DE MÉDIAS ---
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        # 1. Inicializar parâmetros com uma semente diferente por run para não enviesar
        run_key = jax.random.fold_in(main_key, run)
        layer_keys = jax.random.split(run_key, len(width) - 1)
        params = []
        
        for k, lin, lout in zip(layer_keys, width[:-1], width[1:]):
            W = initializer(k, (lin, lout), jnp.float32)
            B = initializer(k, (1, lout), jnp.float32)
            params.append({'W': W, 'B': B})

        opt_state = optimizer.init(params)

        t_start_adam = time.perf_counter()
        for e in range(epochs_adam):
            opt_state, params = update(opt_state, params)
        
        # Para calcular a Loss final do Adam, geramos uma amostra
        sampler_eval = qmc.LatinHypercube(d=1)
        X_eval_adam = jnp.array(sampler_eval.random(n=n), dtype=jnp.float32)
        loss_adam = loss_function(params, X_eval_adam).block_until_ready()
        
        t_adam = time.perf_counter() - t_start_adam
        acc_train_adam += t_adam
        print(f"Adam concluído em {t_adam:.2f}s | Loss: {loss_adam:.8e}")

        # 3. Refinamento L-BFGS
        params_flat, _ = jax.flatten_util.ravel_pytree(params)
        
        # Gera os pontos para o L-BFGS com SciPy, espelhando o que o autor faz
        sampler_lbfgs = qmc.LatinHypercube(d=1)
        X_lbfgs = jnp.array(sampler_lbfgs.random(n=n), dtype=jnp.float32)
        
        t_start_lbfgs = time.perf_counter()
        params_flat, state, loss_traj_run, err_traj_run = lbfgs_segment_with_trajectory(
            params_flat, x_colloc=X_lbfgs
        )
        
        params = unflatten_fn(params_flat).copy() 
        loss_lbfgs = loss_function(params, X_lbfgs).block_until_ready()
        t_lbfgs = time.perf_counter() - t_start_lbfgs
        
        loss_trajectory_run = np.asarray(loss_traj_run)
        l2_error_trajectory_run = np.asarray(err_traj_run)
        all_loss_trajectories.append(loss_trajectory_run)
        all_l2_error_trajectories.append(l2_error_trajectory_run)

        acc_train_lbfgs += t_lbfgs
        print(f"L-BFGS concluído em {t_lbfgs:.2f}s | Loss: {loss_lbfgs:.8e}")

        # 4. Avaliação nos pontos Ground Truth
        t_start_eval = time.perf_counter()
        Y_nn = forward(X_test, params).reshape(X_test.shape)
        Y_nn.block_until_ready()
        t_eval = time.perf_counter() - t_start_eval
        acc_eval_time += t_eval

        # Cálculo do erro relativo para esta run
        norma_erro = jnp.linalg.norm(Y_nn - Y_test)
        norma_exata = jnp.linalg.norm(Y_test)
        rel_l2_error = float(norma_erro / norma_exata)
        l2_errors.append(rel_l2_error)
        print(f"Erro L2 Relativo (Run {run+1}): {rel_l2_error:.8e}") 
        
        if run == num_runs - 1:
            Y_nn_final = np.asarray(Y_nn)
            params_final_flat = np.asarray(params_flat)

    # --- CÁLCULO DAS MÉDIAS E MEDIANAS ---
    avg_train_adam = float(acc_train_adam / num_runs)
    avg_train_lbfgs = float(acc_train_lbfgs / num_runs)
    avg_train_total = float(avg_train_adam + avg_train_lbfgs)
    avg_eval_time = float(acc_eval_time / num_runs)
    
    avg_rel_l2 = float(np.mean(l2_errors))
    median_rel_l2 = float(np.median(l2_errors))

    print("\n" + "-"*20)
    print(f"RESUMO DAS MÉDIAS ({num_runs} RUNS)")
    print(f"Erro L2 Relativo Médio: {avg_rel_l2:.8e}")
    print(f"Tempo Médio Treino Adam: {avg_train_adam:.3f}s")
    print(f"Tempo Médio Treino Total: {avg_train_total:.3f}s")
    print(f"Tempo Médio Avaliação: {avg_eval_time:.5f}s")
    print("-"*20)

    output_dir = Path("pinn_poisson_1d")
    output_dir.mkdir(exist_ok=True)

    # 6. Salvar Dados (.json)
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

    nome_arquivo = output_dir / f'dados_pinn_1d_{arch_str}.json' 
    with open(nome_arquivo, 'w') as f:
        json.dump(results, f, indent=4)
        
    points = {
        'x': X_test.flatten().tolist(),
        'y_nn': Y_nn_final.tolist(),             
        'network_weights': params_final_flat.tolist() 
    }

    nome_arquivo = output_dir / f'pontos_pinn_1d_{arch_str}.json' 
    with open(nome_arquivo, 'w') as f:
        json.dump(points, f, indent=4)

    all_loss_trajectories = np.stack(all_loss_trajectories, axis=0)
    all_l2_error_trajectories = np.stack(all_l2_error_trajectories, axis=0)
    
    training_curve = {
        'architecture': width,
        'method': 'PINN',
        'epochs_adam': epochs_adam,
        'lbfgs_maxiter': lbfgs_maxiter,
        'num_runs': num_runs,
        'steps': list(range(10, lbfgs_maxiter + 1, 10)),
        'l2_relative_error_per_run': all_l2_error_trajectories.tolist(),
        'loss_per_run': all_loss_trajectories.tolist(),
        'l2_relative_error_mean': all_l2_error_trajectories.mean(axis=0).tolist(),
        'l2_relative_error_std': all_l2_error_trajectories.std(axis=0).tolist(),
    }

    nome_arquivo = output_dir / f'curva_treino_pinn_1d_{arch_str}.json'
    with open(nome_arquivo, 'w') as f:
        json.dump(training_curve, f, indent=4)
    
    print(f"Dados salvos como: {nome_arquivo}\n")