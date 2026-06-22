# -*- coding: utf-8 -*-
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import numpy as np
import time
import json

import jax
import jax.numpy as jnp
import optax
import jaxopt

from scipy.stats import qmc

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

# Geração do Ground Truth com malha de 1000 pontos
X_test = jnp.linspace(x_lower, x_upper, 1000).reshape((1000, 1))
Y_test = X_test * jnp.exp(-X_test**2)

print(f"Ground truth gerado analiticamente com {X_test.shape[0]} pontos.")

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
    
    lbfgs = jaxopt.LBFGS(fun=objective_lbfgs, maxiter=50000, history_size=50, tol=1e-8)

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
        params_flat, state = lbfgs.run(params_flat, x_colloc=X_lbfgs)
        
        params = unflatten_fn(params_flat).copy() 
        loss_lbfgs = loss_function(params, X_lbfgs).block_until_ready()
        t_lbfgs = time.perf_counter() - t_start_lbfgs
        
        acc_train_lbfgs += t_lbfgs
        print(f"L-BFGS concluído em {t_lbfgs:.2f}s | Loss: {loss_lbfgs:.8e} ({state.iter_num} iters)")

        # 4. Avaliação nos pontos Ground Truth (Malha de 1000 pontos)
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
    print(f"Erro L2 Relativo Mediana: {median_rel_l2:.8e}")
    print(f"Tempo Médio Treino Adam: {avg_train_adam:.3f}s")
    print(f"Tempo Médio Treino Total: {avg_train_total:.3f}s")
    print(f"Tempo Médio Avaliação: {avg_eval_time:.5f}s")
    print("-"*20)

    # 6. Salvar Dados (.json)
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
        'y_nn': Y_nn_final.tolist(),             
        'network_weights': params_final_flat.tolist() 
    }

    nome_arquivo = f'dados_pinn_1d_{arch_str}.json' 
    with open(nome_arquivo, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Dados salvos como: {nome_arquivo}\n")
    
print("="*20)
print("TODAS AS ARQUITETURAS FORAM TREINADAS E AVALIADAS COM SUCESSO!")
print("="*20)