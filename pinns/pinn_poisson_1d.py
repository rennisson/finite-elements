# -*- coding: utf-8 -*-
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import numpy as np
import time

import jax
import jax.numpy as jnp
import optax
import jaxopt


activation_function = jax.nn.tanh

def u(x):
    """Exact solution of the PDE"""
    return x * jnp.exp(-x**2)

def rhs(x):
    """Defines the right hand side of the equation"""
    return (4*x**3 - 6*x) * jnp.exp(-x**2)

# Dominio de x
x_lower = 0
x_upper = 1
n  = 256
key_x = jax.random.PRNGKey(136)

X = jax.random.uniform(
    key=key_x, shape=(n, 1),
    minval=x_lower, maxval=x_upper
)
X_test = jnp.linspace(x_lower, x_upper, 1000).reshape((1000, 1))
Y_test = X_test * jnp.exp(-X_test**2)

# ===== CONFIGURAÇÕES GERAIS =====
learning_rate = 1e-4
b1 = 0.9
b2 = 0.999
eps = 1e-08
eps_root = 0.0
epochs_adam = 15000

# Inicializa o otimizador Adam
optimizer = optax.adam(learning_rate, b1, b2, eps, eps_root)

# ===== FUNÇÕES JAX =====
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
def update(opt_state, params, x):
    grads = grad_loss(params, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

# ===== LISTA DE ARQUITETURAS =====
architectures = [
    [1, 1], [2, 1], [5, 1], [10, 1], [20, 1], [40, 1], 
    [5, 5, 1], [10, 10, 1], [20, 20, 1], [40, 40, 1], 
    [5, 5, 5, 1], [10, 10, 10, 1], [20, 20, 20, 1], [40, 40, 40, 1]
]

# ===== LOOP PRINCIPAL DE TREINAMENTO =====
for arch in architectures:
    width = [1] + arch
    arch_str = "_".join(map(str, width)) # Ex: "1_5_5_1" para salvar os nomes dos arquivos
    
    print("\n" + "="*80)
    print(f"INICIANDO TREINAMENTO DA ARQUITETURA: {width}")
    print("="*80)

    # 1. Inicializar parâmetros para a arquitetura atual
    initializer = jax.nn.initializers.glorot_normal()
    key = jax.random.split(jax.random.PRNGKey(136), len(width) - 1)
    params = []
    
    for k, lin, lout in zip(key, width[:-1], width[1:]):
        W = initializer(k, (lin, lout), jnp.float32)
        B = initializer(k, (1, lout), jnp.float32)
        params.append({'W': W, 'B': B})

    # Inicializar o estado do otimizador
    opt_state = optimizer.init(params)

    # 2. Treinamento ADAM
    print("-> Iniciando Adam...")
    t_start_train = time.time()
    for e in range(epochs_adam):
        opt_state, params = update(opt_state, params, X)
        
        if e % 5000 == 0:
            elapsed = time.time() - t_start_train
            print(f"   {elapsed:.2f}s | Adam Epoch {e:<5}: emp. loss {loss_function(params, X):.12f}")

    loss_adam = loss_function(params, X).block_until_ready()
    train_time_adam = time.time() - t_start_train
    print(f"-> Adam concluído. Loss: {loss_adam:.12f}")

    # 3. Refinamento L-BFGS
    print("\n-> Iniciando L-BFGS...")
    params_flat, unflatten_fn = jax.flatten_util.ravel_pytree(params)

    @jax.jit
    def objective_lbfgs(p_flat):
        return loss_function(unflatten_fn(p_flat), X)

    t0_lbfgs = time.time()
    lbfgs = jaxopt.LBFGS(fun=objective_lbfgs, maxiter=15000, history_size=50, tol=1e-8)
    params_final_flat, state = lbfgs.run(params_flat)
    elapsed_lbfgs = time.time() - t0_lbfgs

    params = unflatten_fn(params_final_flat)
    loss_lbfgs = loss_function(params, X)
    num_iters = state.iter_num

    print(f"-> L-BFGS concluído. Loss: {loss_lbfgs:.12f} ({num_iters} iterações)")

    # 4. Avaliação
    t_start_eval = time.time()
    Y_nn = forward(X_test, params).reshape(X_test.shape)
    Y_nn.block_until_ready()
    eval_time = time.time() - t_start_eval

    norma_erro = jnp.linalg.norm(Y_nn - Y_test)
    norma_exata = jnp.linalg.norm(Y_test)
    rel_l2_error = float(norma_erro / norma_exata)

    # 5. Resumo Rápido
    print("\n--- RESUMO ---")
    print(f"Erro L2 Relativo: {rel_l2_error:.8e}")
    print(f"Melhoria de Loss com L-BFGS: {(loss_adam - loss_lbfgs) / loss_adam * 100:.2f}%")

    # 6. Salvar e Gerar Gráficos Silenciosamente
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico 1: Solução
    ax[0].plot(X_test, Y_test, '-', color='blue', markersize=0.5, label='Exact solution')
    ax[0].plot(X_test, Y_nn, '--', color='red', markersize=0.5, label=f'PINN {width}')
    ax[0].set_title('Aproximação vs Exata')
    ax[0].legend(loc='best')

    plt.tight_layout()
    plt.savefig(f'plot_pinn_{arch_str}.png', dpi=150)
    plt.close() # Importante: fecha o gráfico para liberar a memória ram

    # 7. Salvar Dados (.npz)
    num_params = len(params_final_flat)
    num_hidden_layers = len(width) - 2

    results = {
        'architecture': width,
        'num_hidden_layers': int(num_hidden_layers),
        'epochs_adam': epochs_adam,
        'error_relativo': rel_l2_error,
        'time_training': float(train_time_adam),
        'time_training_w_lbfgs': float(train_time_adam + elapsed_lbfgs),
        'time_evaluation': float(eval_time),
        'num_params': int(num_params),
        'y_nn': np.asarray(Y_nn),
        'network_weights': np.asarray(params_final_flat)
    }

    nome_arquivo = f'dados_pinn_lbfgs_1d_{arch_str}.npz'
    np.savez_compressed(nome_arquivo, **results)
    
    print(f"Gráfico salvo como: plot_pinn_{arch_str}.png")
    print(f"Dados salvos como: {nome_arquivo}\n")
    
print("="*80)
print("TODAS AS ARQUITETURAS FORAM TREINADAS E TESTADAS COM SUCESSO!")
print("="*80)