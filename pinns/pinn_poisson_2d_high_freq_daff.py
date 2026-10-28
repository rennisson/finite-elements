# -*- coding: utf-8 -*-
import os
import sys

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

a = 1

architectures = [
    [264, 264, 1],
    [264, 264, 264, 1],
    [512, 512, 1],
    [512, 512, 512, 1]
]

activation_function = jax.nn.tanh
lr = 1e-3
epochs_adam = 0
lbfgs_maxiter = 5000
eval_freq = 20
num_runs = 3
main_key = jax.random.PRNGKey(42)

N_f = 2000
N_g = 250

def u(x, y, a=a):
    """Solução exata da EDP (Eq. 6.4)"""
    return (
        jnp.sin(jnp.pi * x)
        * jnp.sin(jnp.pi * y)
        * (jnp.sin(a * (x + y)) + jnp.sin(2.0 * jnp.pi * x) + jnp.cos(3.0 * jnp.pi * y))
    )


def pde_rhs(x, y, a=a):
    """
    Define o lado direito para Delta u = u_xx + u_yy (Eq. 6.4: -Delta u = f, logo Delta u = -f),
    calculado analiticamente a partir de:
        u(x, y) = sin(pi * x) * sin(pi * y) * [sin(a * (x + y)) + sin(2 * pi * x) + cos(3 * pi * y)]
    """
    sin_px = jnp.sin(jnp.pi * x)
    cos_px = jnp.cos(jnp.pi * x)
    sin_py = jnp.sin(jnp.pi * y)
    cos_py = jnp.cos(jnp.pi * y)

    sin_axy = jnp.sin(a * (x + y))
    cos_axy = jnp.cos(a * (x + y))
    sin_2px = jnp.sin(2.0 * jnp.pi * x)
    cos_2px = jnp.cos(2.0 * jnp.pi * x)
    sin_3py = jnp.sin(3.0 * jnp.pi * y)
    cos_3py = jnp.cos(3.0 * jnp.pi * y)

    pi = jnp.pi
    pi2 = pi ** 2

    term1 = sin_px * sin_py * (
        -2.0 * (pi2 + a ** 2) * sin_axy
        - 6.0 * pi2 * sin_2px
        - 11.0 * pi2 * cos_3py
    )
    term2 = 2.0 * pi * cos_px * sin_py * (a * cos_axy + 2.0 * pi * cos_2px)
    term3 = 2.0 * pi * sin_px * cos_py * (a * cos_axy - 3.0 * pi * sin_3py)
    return term1 + term2 + term3


# ==========================================
# 1. CARREGAMENTO DO GROUND TRUTH E PDE
# ==========================================
gt_filename = f"gt_poisson_2d_high_freq_a{a}.json"
gt_candidates = [
    Path(gt_filename),
    Path(__file__).resolve().parent / gt_filename,
    Path(__file__).resolve().parent.parent / "sv-pinns" / gt_filename,
]
gt_path = next((p for p in gt_candidates if p.exists()), None)

if gt_path is None:
    gt_dir = str(Path(__file__).resolve().parent)
    if gt_dir not in sys.path:
        sys.path.insert(0, gt_dir)
    from ground_truth_poisson_2d_high_freq import generate_analytical_solution_2d

    gt_path = Path(gt_filename)
    generate_analytical_solution_2d(a=a, filename=str(gt_path))

with open(gt_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

X_mesh_gt = jnp.array(gt_data["X"])
Y_mesh_gt = jnp.array(gt_data["Y"])
U_true_gt = jnp.array(gt_data["U_true"])

# Pontos e valores de referência achatados, usados nas avaliações de trajetória
xy_points_gt = jnp.stack([X_mesh_gt.ravel(), Y_mesh_gt.ravel()], axis=-1)
U_true_flat = U_true_gt.ravel()

print(f"Ground truth '{gt_path}' carregado (a = {a}). Shape da malha de avaliação: {X_mesh_gt.shape}")

# ==========================================
# 2. DEFINIÇÕES GERAIS DA REDE, DAFF E OTIMIZADOR
# ==========================================
optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-08)

# ==========================================
# 2.1. DAFF (DOMAIN-AWARE FOURIER FEATURES) EM 2D
#      Codificação DAFF (Sec. 4.1.1, Eq. 4.2, d=2):
#          phi_{k1, k2}(x, y) = 2 * sin(k1 * pi * x) * sin(k2 * pi * y)
#      Selecionam-se as n_daff = 64 * d = 128 autofunções do Laplaciano de
#      Dirichlet em (0, 1)^2 com os menores autovalores lambda_k = pi^2 (k1^2 + k2^2)
#      dentre k1, k2 <= 64 (Sec. 6 / Table A.8).
# ==========================================
n_daff = 128  # número de funções de base DAFF (Table A.8, caso 2D: 64 * d = 128)


def build_daff_frequencies_2d(n_daff=n_daff, k_max=64):
    """
    Seleciona os `n_daff` pares de frequências (k1, k2) in {1, ..., k_max}^2
    associados aos menores autovalores lambda_{k1, k2} = pi^2 * (k1^2 + k2^2)
    do Laplaciano de Dirichlet em (0, 1)^2 (Sec. 4.1.1 e Sec. 6).
    """
    k_vals = np.arange(1, k_max + 1, dtype=np.int64)
    k1_grid, k2_grid = np.meshgrid(k_vals, k_vals, indexing="ij")
    pairs = np.stack([k1_grid.ravel(), k2_grid.ravel()], axis=-1)
    eig_proxy = pairs[:, 0] ** 2 + pairs[:, 1] ** 2
    order = np.argsort(eig_proxy, kind="stable")[:n_daff]
    return jnp.array(pairs[order], dtype=jnp.float64)


DAFF_K_PAIRS = build_daff_frequencies_2d(n_daff=n_daff)


def daff_encoding(xy, k_pairs=DAFF_K_PAIRS):
    """
    Mapeia as coordenadas (x, y) in (0, 1)^2 para o vetor de características DAFF
    pelas autofunções do Laplaciano de Dirichlet em (0, 1)^2 (Eq. 4.2, d=2):
        phi_{k1, k2}(x, y) = 2 * sin(k1 * pi * x) * sin(k2 * pi * y).

    Parametros
    ----------
    xy: tensor de entrada, shape (batch, 2)
    k_pairs: pares de frequências (k1, k2), shape (n_daff, 2)

    Retorna
    ----------
    phi_I(x, y): tensor codificado, shape (batch, n_daff)
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    k1 = k_pairs[:, 0]
    k2 = k_pairs[:, 1]
    return 2.0 * jnp.sin(jnp.pi * x * k1) * jnp.sin(jnp.pi * y * k2)


@jax.jit
def forward(xy, params):
    x = daff_encoding(xy)
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
        return (laplacian_u(x, y, params) - pde_rhs(x, y, a=a)) ** 2
    return jnp.mean(jax.vmap(residual)(xy_interior))


# Condições de contorno de Dirichlet homogêneas em toda a fronteira dOmega (Eq. 6.4: u = 0 em dOmega)
def loss_bc_x0(params, xy): return jnp.mean(jax.vmap(lambda pt: u_net(pt[0], pt[1], params) ** 2)(xy))
def loss_bc_x1(params, xy): return jnp.mean(jax.vmap(lambda pt: u_net(pt[0], pt[1], params) ** 2)(xy))
def loss_bc_y0(params, xy): return jnp.mean(jax.vmap(lambda pt: u_net(pt[0], pt[1], params) ** 2)(xy))
def loss_bc_y1(params, xy): return jnp.mean(jax.vmap(lambda pt: u_net(pt[0], pt[1], params) ** 2)(xy))


def compute_boundary_losses(params, xy_x0, xy_x1, xy_y0, xy_y1):
    """Retorna os resíduos quadráticos médios (MSE) de cada condição de fronteira da Eq. (6.4):
    x=0 (Dirichlet), x=1 (Dirichlet), y=0 (Dirichlet), y=1 (Dirichlet)."""
    return (
        loss_bc_x0(params, xy_x0),
        loss_bc_x1(params, xy_x1),
        loss_bc_y0(params, xy_y0),
        loss_bc_y1(params, xy_y1),
    )


def print_boundary_losses(params, xy_x0, xy_x1, xy_y0, xy_y1):
    """Perda MSE de cada condição de fronteira separadamente (Eq. 6.4), para comparação rápida."""
    loss_x0, loss_x1, loss_y0, loss_y1 = compute_boundary_losses(params, xy_x0, xy_x1, xy_y0, xy_y1)
    labels = ["x=0 (Dirichlet)", "x=1 (Dirichlet)", "y=0 (Dirichlet)", "y=1 (Dirichlet)"]
    losses = [loss_x0, loss_x1, loss_y0, loss_y1]
    print("  Perda residual por condicao de fronteira:")
    for label, val in zip(labels, losses):
        print(f"    {label:<18s}: {float(val):.6e}")


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

output_dir = Path(f"pinn_daff_poisson_2d_high_freq_a{a}")
output_dir.mkdir(exist_ok=True)

for arch in architectures:
    width = [2] + arch          # Dimensão original de entrada (2) para identificação da arquitetura
    layer_dims = [n_daff] + arch  # Primeira camada oculta recebe as `n_daff` características DAFF
    arch_str = "_".join(map(str, width))

    print("\n" + "=" * 50)
    print(f"INICIANDO: ARQUITETURA {width} (DAFF={n_daff}) | a = {a} | {num_runs} EXECUÇÕES")
    print("=" * 50)

    acc_train_adam = 0.0
    acc_train_lbfgs = 0.0
    acc_eval_time = 0.0
    l2_errors = []

    all_loss_trajectories = []           # (num_runs, lbfgs_maxiter // eval_freq)
    all_l2_error_trajectories = []       # (num_runs, lbfgs_maxiter // eval_freq)
    interior_mse_runs = []               # trajetoria MSE residual do interior (L-BFGS)
    bc_x0_runs = []                      # trajetoria MSE contorno x=0 (L-BFGS)
    bc_x1_runs = []                      # trajetoria MSE contorno x=1 (L-BFGS)
    bc_y0_runs = []                      # trajetoria MSE contorno y=0 (L-BFGS)
    bc_y1_runs = []                      # trajetoria MSE contorno y=1 (L-BFGS)

    all_adam_loss_trajectories = []      # (num_runs, epochs_adam // eval_freq)
    all_adam_l2_error_trajectories = []  # (num_runs, epochs_adam // eval_freq)
    adam_interior_mse_runs = []          # trajetoria MSE residual do interior (Adam)
    adam_bc_x0_runs = []                 # trajetoria MSE contorno x=0 (Adam)
    adam_bc_x1_runs = []                 # trajetoria MSE contorno x=1 (Adam)
    adam_bc_y0_runs = []                 # trajetoria MSE contorno y=0 (Adam)
    adam_bc_y1_runs = []                 # trajetoria MSE contorno y=1 (Adam)

    U_nn_final = None
    params_final_flat = None

    # Configuração do descompactador (unflatten_fn) para a arquitetura atual (com entrada DAFF)
    initializer = jax.nn.initializers.glorot_normal()
    dummy_key = jax.random.split(main_key, len(layer_dims) - 1)
    dummy_params = []
    for k, lin, lout in zip(dummy_key, layer_dims[:-1], layer_dims[1:]):
        dummy_params.append({'W': jnp.zeros((lin, lout), dtype=jnp.float64),
                             'B': jnp.zeros((1, lout), dtype=jnp.float64)})

    _, unflatten_fn = jax.flatten_util.ravel_pytree(dummy_params)

    @jax.jit
    def objective_lbfgs(p_flat, xy_interior, xy_x0, xy_x1, xy_y0, xy_y1):
        return loss_function(unflatten_fn(p_flat), xy_interior, xy_x0, xy_x1, xy_y0, xy_y1)

    lbfgs = jaxopt.LBFGS(fun=objective_lbfgs, maxiter=lbfgs_maxiter, history_size=200, tol=1e-12)

    # Captura o erro L2, MSE do interior e perdas de contorno a cada eval_freq passos de L-BFGS
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

            # MSE residual do interior (EDP) e de cada condicao de contorno
            interior_mse = loss_function_interior(params_, xy_interior)
            bc_x0, bc_x1, bc_y0, bc_y1 = compute_boundary_losses(params_, xy_x0, xy_x1, xy_y0, xy_y1)

            return (p_next, state_next), (state_next.value, l2_err, interior_mse, bc_x0, bc_x1, bc_y0, bc_y1)

        (p_final, state_final), (
            loss_traj, err_traj, interior_mse_traj, bc_x0_traj, bc_x1_traj, bc_y0_traj, bc_y1_traj
        ) = jax.lax.scan(block_fn, (p_flat, init_state), xs=None, length=num_blocks)

        return (p_final, state_final, loss_traj, err_traj,
                interior_mse_traj, bc_x0_traj, bc_x1_traj, bc_y0_traj, bc_y1_traj)

    # --- LOOP DE EXECUÇÕES (RUNS) ---
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")

        # 1. Inicializar parâmetros com semente única
        run_key = jax.random.fold_in(main_key, run)
        layer_keys = jax.random.split(run_key, len(layer_dims) - 1)
        params = []
        for k, lin, lout in zip(layer_keys, layer_dims[:-1], layer_dims[1:]):
            W = initializer(k, (lin, lout), jnp.float64)
            B = initializer(k, (1, lout), dtype=jnp.float64)
            params.append({'W': W, 'B': B})

        opt_state = optimizer.init(params)

        adam_loss_traj_run = []
        adam_l2_traj_run = []
        adam_interior_mse_traj_run = []
        adam_bc_x0_traj_run = []
        adam_bc_x1_traj_run = []
        adam_bc_y0_traj_run = []
        adam_bc_y1_traj_run = []

        # 2. Treinamento Adam
        t_start_adam = time.perf_counter()
        for e in range(epochs_adam):
            xy_int, x0, x1, y0, y1 = generate_lhs_points(N_f, N_g)
            opt_state, params = update(opt_state, params, xy_int, x0, x1, y0, y1)

            if (e + 1) % eval_freq == 0:
                u_nn_step = forward(xy_points_gt, params).flatten()
                l2_err_step = jnp.linalg.norm(u_nn_step - U_true_flat) / jnp.linalg.norm(U_true_flat)
                interior_mse_step = loss_function_interior(params, xy_int)
                bc_x0_step, bc_x1_step, bc_y0_step, bc_y1_step = compute_boundary_losses(params, x0, x1, y0, y1)
                loss_step = interior_mse_step + bc_x0_step + bc_x1_step + bc_y0_step + bc_y1_step
                adam_loss_traj_run.append(float(loss_step))
                adam_l2_traj_run.append(float(l2_err_step))
                adam_interior_mse_traj_run.append(float(interior_mse_step))
                adam_bc_x0_traj_run.append(float(bc_x0_step))
                adam_bc_x1_traj_run.append(float(bc_x1_step))
                adam_bc_y0_traj_run.append(float(bc_y0_step))
                adam_bc_y1_traj_run.append(float(bc_y1_step))

        t_adam = time.perf_counter() - t_start_adam
        acc_train_adam += t_adam

        # Avaliar loss final do adam para logs
        xy_int, x0, x1, y0, y1 = generate_lhs_points(N_f, N_g)
        loss_adam = loss_function(params, xy_int, x0, x1, y0, y1).block_until_ready()
        print(f"Adam concluído em {t_adam:.2f}s | Loss: {loss_adam:.8e}")

        all_adam_loss_trajectories.append(np.asarray(adam_loss_traj_run))
        all_adam_l2_error_trajectories.append(np.asarray(adam_l2_traj_run))
        adam_interior_mse_runs.append(np.asarray(adam_interior_mse_traj_run))
        adam_bc_x0_runs.append(np.asarray(adam_bc_x0_traj_run))
        adam_bc_x1_runs.append(np.asarray(adam_bc_x1_traj_run))
        adam_bc_y0_runs.append(np.asarray(adam_bc_y0_traj_run))
        adam_bc_y1_runs.append(np.asarray(adam_bc_y1_traj_run))

        # 3. Refinamento L-BFGS
        params_flat, _ = jax.flatten_util.ravel_pytree(params)

        xy_int_lbfgs, x0_lbfgs, x1_lbfgs, y0_lbfgs, y1_lbfgs = generate_lhs_points(N_f, N_g)

        t_start_lbfgs = time.perf_counter()
        (
            params_flat, state, loss_traj_run, err_traj_run,
            interior_mse_traj_run, bc_x0_traj_run, bc_x1_traj_run, bc_y0_traj_run, bc_y1_traj_run
        ) = lbfgs_segment_with_trajectory(
            params_flat, xy_int_lbfgs, x0_lbfgs, x1_lbfgs, y0_lbfgs, y1_lbfgs
        )
        t_lbfgs = time.perf_counter() - t_start_lbfgs
        acc_train_lbfgs += t_lbfgs

        params = unflatten_fn(params_flat).copy()
        loss_lbfgs = loss_function(params, xy_int_lbfgs, x0_lbfgs, x1_lbfgs, y0_lbfgs, y1_lbfgs).block_until_ready()
        print(f"L-BFGS concluído em {t_lbfgs:.2f}s | Loss: {loss_lbfgs:.8e}")

        loss_trajectory_run = np.asarray(loss_traj_run)
        l2_error_trajectory_run = np.asarray(err_traj_run)
        run_interior_mse_traj = np.asarray(interior_mse_traj_run)
        run_bc_x0_traj = np.asarray(bc_x0_traj_run)
        run_bc_x1_traj = np.asarray(bc_x1_traj_run)
        run_bc_y0_traj = np.asarray(bc_y0_traj_run)
        run_bc_y1_traj = np.asarray(bc_y1_traj_run)

        all_loss_trajectories.append(loss_trajectory_run)
        all_l2_error_trajectories.append(l2_error_trajectory_run)
        interior_mse_runs.append(run_interior_mse_traj)
        bc_x0_runs.append(run_bc_x0_traj)
        bc_x1_runs.append(run_bc_x1_traj)
        bc_y0_runs.append(run_bc_y0_traj)
        bc_y1_runs.append(run_bc_y1_traj)

        print(f"  MSE residual do interior: {float(run_interior_mse_traj[-1]):.6e}")
        print_boundary_losses(params, x0_lbfgs, x1_lbfgs, y0_lbfgs, y1_lbfgs)

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
        print(f"Erro L2 Relativo (Run {run + 1}): {rel_l2_error:.8e}")

        if run == num_runs - 1:
            U_nn_final = np.asarray(u_nn)
            params_final_flat = np.asarray(params_flat)

    # --- CÁLCULO DAS MÉDIAS E EXPORTAÇÃO ---
    avg_train_adam = float(acc_train_adam / num_runs)
    avg_train_lbfgs = float(acc_train_lbfgs / num_runs)
    avg_train_total = float(avg_train_adam + avg_train_lbfgs)  # Representa tempo acumulado
    avg_eval_time = float(acc_eval_time / num_runs)

    avg_rel_l2 = float(np.mean(l2_errors))
    median_rel_l2 = float(np.median(l2_errors))

    print("\n" + "-" * 30)
    print(f"RESUMO DAS MÉDIAS ({num_runs} RUNS) - ARQUITETURA {width}")
    print(f"Erro L2 Relativo Médio: {avg_rel_l2:.8e}")
    print(f"Tempo Médio Treino Total: {avg_train_total:.3f}s")
    print("-" * 30)

    # 5. Salvar Dados (.json) - mesma estrutura de 3 arquivos usada no 2D/1D
    results = {
        'architecture': width,
        'num_hidden_layers': len(width) - 1,
        'features': 'DAFF',
        'n_daff': n_daff,
        'a': a,
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

    nome_arquivo = output_dir / f'dados_pinn_2d_a{a}_{arch_str}.json'
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    points = {
        'x': X_mesh_gt.ravel().tolist(),
        'y': Y_mesh_gt.ravel().tolist(),
        'u_nn': U_nn_final.ravel().tolist(),
        'network_weights': params_final_flat.tolist()
    }

    nome_arquivo = output_dir / f'pontos_pinn_2d_a{a}_{arch_str}.json'
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(points, f, indent=4)

    all_loss_trajectories = np.stack(all_loss_trajectories, axis=0)
    all_l2_error_trajectories = np.stack(all_l2_error_trajectories, axis=0)
    all_interior_mse_trajectories = np.stack(interior_mse_runs, axis=0)
    all_bc_x0_trajectories = np.stack(bc_x0_runs, axis=0)
    all_bc_x1_trajectories = np.stack(bc_x1_runs, axis=0)
    all_bc_y0_trajectories = np.stack(bc_y0_runs, axis=0)
    all_bc_y1_trajectories = np.stack(bc_y1_runs, axis=0)

    all_adam_loss_trajectories = np.stack(all_adam_loss_trajectories, axis=0)
    all_adam_l2_error_trajectories = np.stack(all_adam_l2_error_trajectories, axis=0)
    all_adam_interior_mse_trajectories = np.stack(adam_interior_mse_runs, axis=0)
    all_adam_bc_x0_trajectories = np.stack(adam_bc_x0_runs, axis=0)
    all_adam_bc_x1_trajectories = np.stack(adam_bc_x1_runs, axis=0)
    all_adam_bc_y0_trajectories = np.stack(adam_bc_y0_runs, axis=0)
    all_adam_bc_y1_trajectories = np.stack(adam_bc_y1_runs, axis=0)

    training_curve = {
        'architecture': width,
        'method': 'PINN',
        'features': 'DAFF',
        'n_daff': n_daff,
        'a': a,
        'epochs_adam': epochs_adam,
        'lbfgs_maxiter': lbfgs_maxiter,
        'num_runs': num_runs,
        'steps_adam': list(range(eval_freq, epochs_adam + 1, eval_freq)),
        'l2_relative_error_per_run_adam': all_adam_l2_error_trajectories.tolist(),
        'loss_per_run_adam': all_adam_loss_trajectories.tolist(),
        'l2_relative_error_mean_adam': all_adam_l2_error_trajectories.mean(axis=0).tolist(),
        'l2_relative_error_std_adam': all_adam_l2_error_trajectories.std(axis=0).tolist(),
        # Interior e Condições de contorno (Adam)
        'interior_loss_per_run_adam': all_adam_interior_mse_trajectories.tolist(),
        'interior_loss_mean_adam': all_adam_interior_mse_trajectories.mean(axis=0).tolist(),
        'interior_loss_std_adam': all_adam_interior_mse_trajectories.std(axis=0).tolist(),
        'bc_loss_x0_per_run_adam': all_adam_bc_x0_trajectories.tolist(),
        'bc_loss_x0_mean_adam': all_adam_bc_x0_trajectories.mean(axis=0).tolist(),
        'bc_loss_x0_std_adam': all_adam_bc_x0_trajectories.std(axis=0).tolist(),
        'bc_loss_x1_per_run_adam': all_adam_bc_x1_trajectories.tolist(),
        'bc_loss_x1_mean_adam': all_adam_bc_x1_trajectories.mean(axis=0).tolist(),
        'bc_loss_x1_std_adam': all_adam_bc_x1_trajectories.std(axis=0).tolist(),
        'bc_loss_y0_per_run_adam': all_adam_bc_y0_trajectories.tolist(),
        'bc_loss_y0_mean_adam': all_adam_bc_y0_trajectories.mean(axis=0).tolist(),
        'bc_loss_y0_std_adam': all_adam_bc_y0_trajectories.std(axis=0).tolist(),
        'bc_loss_y1_per_run_adam': all_adam_bc_y1_trajectories.tolist(),
        'bc_loss_y1_mean_adam': all_adam_bc_y1_trajectories.mean(axis=0).tolist(),
        'bc_loss_y1_std_adam': all_adam_bc_y1_trajectories.std(axis=0).tolist(),
        'steps': list(range(eval_freq, lbfgs_maxiter + 1, eval_freq)),
        'l2_relative_error_per_run': all_l2_error_trajectories.tolist(),
        'loss_per_run': all_loss_trajectories.tolist(),
        'l2_relative_error_mean': all_l2_error_trajectories.mean(axis=0).tolist(),
        'l2_relative_error_std': all_l2_error_trajectories.std(axis=0).tolist(),
        # Interior (MSE do resíduo da EDP)
        'interior_loss_per_run': all_interior_mse_trajectories.tolist(),
        'interior_loss_mean': all_interior_mse_trajectories.mean(axis=0).tolist(),
        'interior_loss_std': all_interior_mse_trajectories.std(axis=0).tolist(),
        # Condições de contorno (MSE em x=0, x=1, y=0 e y=1)
        'bc_loss_x0_per_run': all_bc_x0_trajectories.tolist(),
        'bc_loss_x0_mean': all_bc_x0_trajectories.mean(axis=0).tolist(),
        'bc_loss_x0_std': all_bc_x0_trajectories.std(axis=0).tolist(),
        'bc_loss_x1_per_run': all_bc_x1_trajectories.tolist(),
        'bc_loss_x1_mean': all_bc_x1_trajectories.mean(axis=0).tolist(),
        'bc_loss_x1_std': all_bc_x1_trajectories.std(axis=0).tolist(),
        'bc_loss_y0_per_run': all_bc_y0_trajectories.tolist(),
        'bc_loss_y0_mean': all_bc_y0_trajectories.mean(axis=0).tolist(),
        'bc_loss_y0_std': all_bc_y0_trajectories.std(axis=0).tolist(),
        'bc_loss_y1_per_run': all_bc_y1_trajectories.tolist(),
        'bc_loss_y1_mean': all_bc_y1_trajectories.mean(axis=0).tolist(),
        'bc_loss_y1_std': all_bc_y1_trajectories.std(axis=0).tolist(),
    }

    nome_arquivo = output_dir / f'curva_treino_pinn_2d_a{a}_{arch_str}.json'
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(training_curve, f, indent=4)

    print(f"Dados salvos como: {nome_arquivo}\n")
