# -*- coding: utf-8 -*-
"""
SV-PINNs: Experimento 1 (Secao 6.1) -- Poisson multi-escala/alta-frequencia em 1D.

    u_xx(x) = -4 pi^2 sin(2 pi x) - 0.1 (a pi)^2 sin(a pi x),  x in (0,1)
    u(0) = u(1) = 0                                                    (Eq. 6.3)

Solucao exata:
    u(x) = sin(2 pi x) + 0.1 sin(a pi x)

Reaproveita a infraestrutura de amostragem de funcoes teste (DST-I, Sec.
4.3.1) e da norma Phi-estocasticamente fraca empirica (Eq. 4.4) de
`1d_poisson.py`. A rede e' a modified-MLP com conexoes residuais
multiplicativas (Sec. 4.1, Wang et al. 2021) alimentada por Domain-Aware
Fourier Features (DAFF, Sec. 4.1.1), que impoe a condicao de contorno de
Dirichlet homogenea de forma "hard" (vieses fixados em zero). O SV-PINN e'
treinado minimizando (4.3) com lambda = 0 (Eq. 4.3), via L-BFGS.
"""
import jax
import jax.flatten_util
import jax.numpy as jnp
import jaxopt
import json
import numpy as np
import os
import time

from jax import random, config

config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

activation_function = jnp.tanh

# ==========================================
# 0. PROBLEMA: Poisson multi-escala 1D (Eq. 6.3)
# ==========================================
def make_problem(a):
    """Retorna (u_exata, rhs) para o parametro de frequencia a (Eq. 6.3)."""
    def u_exact(x):
        return jnp.sin(2 * jnp.pi * x) + 0.1 * jnp.sin(a * jnp.pi * x)

    def rhs(x):
        return (-4 * jnp.pi ** 2 * jnp.sin(2 * jnp.pi * x)
                - 0.1 * (a * jnp.pi) ** 2 * jnp.sin(a * jnp.pi * x))

    return u_exact, rhs


# ==========================================
# 1. DST-I (ortonormal, auto-inversa) -- Sec. 4.3.1
# ==========================================
def dst1_matrix(n, dtype=jnp.float64):
    h = 1.0 / (n + 1)
    idx = jnp.arange(1, n + 1, dtype=dtype)
    return jnp.sqrt(2 * h) * jnp.sin(jnp.pi * jnp.outer(idx, idx) * h)


def dst1_nd(u, S):
    out = u
    for axis in range(u.ndim):
        out = jnp.moveaxis(jnp.tensordot(S, out, axes=([1], [axis])), 0, axis)
    return out


def discrete_eigenvalues(n, d, dtype=jnp.float64):
    h = 1.0 / (n + 1)
    k = jnp.arange(1, n + 1, dtype=dtype)
    lam_1d = (4.0 / (h ** 2)) * (jnp.sin(jnp.pi * k * h / 2.0) ** 2)
    grids = jnp.meshgrid(*([lam_1d] * d), indexing="ij")
    return sum(grids)


def sample_test_functions(key, n, d, N, tau=1.0, dtype=jnp.float64):
    """N realizacoes iid de Phi^(h) (Eq. 4.10), shape (N,) + (n,)*d."""
    S = dst1_matrix(n, dtype=dtype)
    lam_k = discrete_eigenvalues(n, d, dtype=dtype)
    weight = (1.0 + lam_k) ** (-0.5)
    h = 1.0 / (n + 1)
    scale = (h ** (-d / 2.0)) * tau

    keys = random.split(key, N)

    def _one(k):
        W = random.normal(k, (n,) * d, dtype=dtype)
        Phi = dst1_nd(weight * dst1_nd(W, S), S)
        return scale * Phi

    return jax.vmap(_one)(keys)


def grid_points(n, d, dtype=jnp.float64):
    h = 1.0 / (n + 1)
    coords = (jnp.arange(1, n + 1, dtype=dtype) * h,) * d
    mesh = jnp.meshgrid(*coords, indexing="ij")
    return jnp.stack(mesh, axis=-1)


# ==========================================
# 2. Norma Phi-estocasticamente fraca empirica (Eq. 4.4)
# ==========================================
def sv_pinn_norm_squared(residual, phi_samples):
    axes = tuple(range(1, phi_samples.ndim))
    inner_products = jnp.mean(residual[None, ...] * phi_samples, axis=axes)
    return jnp.mean(inner_products ** 2)


# ==========================================
# 3. DAFF -- Domain-Aware Fourier Features (Sec. 4.1.1, Eq. 4.2, d=1)
#    phi_k(x) = sqrt(2) sin(k pi x), k = 1, ..., n_daff
# ==========================================
def daff_encoding(x, n_daff):
    """x: (batch, 1) -> (batch, n_daff)."""
    k = jnp.arange(1, n_daff + 1, dtype=x.dtype)
    phi = jnp.sqrt(2.0) * jnp.sin(jnp.pi * k * x)
    on_boundary = (x == 0.0) | (x == 1.0)
    return jnp.where(on_boundary, 0.0, phi)


# ==========================================
# 4. Modified MLP com conexoes residuais multiplicativas (Sec. 4.1)
#    Vieses fixados em zero: como phi_k(0) = phi_k(1) = 0, a rede satisfaz
#    u_theta(0) = u_theta(1) = 0 automaticamente (hard constraint, Sec. 4.1.1).
# ==========================================
def init_modified_mlp(key, n_daff, width, depth):
    """
    Parametros
    ----------
    n_daff: dimensao da codificacao DAFF (entrada da rede), m
    width: numero de nos por camada oculta, r
    depth: numero de camadas ocultas, K

    Retorna
    ----------
    Dicionario de parametros (sem vieses, cf. Sec. 4.1).
    """
    initializer = jax.nn.initializers.glorot_normal()
    keys = random.split(key, depth + 3)

    params = {
        'W1': initializer(keys[0], (n_daff, width), jnp.float64),   # encoder de U(x)
        'W2': initializer(keys[1], (n_daff, width), jnp.float64),   # encoder de V(x)
        'hidden': [],
    }
    in_dim = n_daff
    for l in range(depth):
        params['hidden'].append(initializer(keys[2 + l], (in_dim, width), jnp.float64))
        in_dim = width
    params['Wout'] = initializer(keys[-1], (width, 1), jnp.float64)
    return params


def modified_mlp_forward(x, params):
    """
    x: (batch, 1) -> u_theta(x): (batch, 1).

    g^(0)(x) = phi_I(x)                         (DAFF)
    U(x) = sigma(W1 g^(0)(x)),  V(x) = sigma(W2 g^(0)(x))
    f^(l)(x) = sigma(W^(l) g^(l-1)(x))
    g^(l)(x) = f^(l)(x) * U(x) + (1 - f^(l)(x)) * V(x)
    u_theta(x) = W^(K+1) g^(K)(x)
    """
    n_daff = params['W1'].shape[0]
    g0 = daff_encoding(x, n_daff)

    U = activation_function(g0 @ params['W1'])
    V = activation_function(g0 @ params['W2'])

    for W in params['hidden']:
        f = activation_function(g0 @ W)
        g0 = f * U + (1.0 - f) * V
    return g0 @ params['Wout']


# ==========================================
# 5. Residuo da EDP (Eq. 6.3), via diferenciacao automatica
# ==========================================
def laplacian_1d(f):
    return jax.grad(jax.grad(f))


def residual_interior(params, x_grid, rhs_fn):
    """R(x_c^(i)) = u_theta_xx(x_c^(i)) - rhs(x_c^(i)) no grid de colocacao."""
    def u_net(x_val):
        x_v = x_val.reshape(1, 1)
        return modified_mlp_forward(x_v, params)[0, 0]

    u_xx = jax.vmap(laplacian_1d(u_net))(x_grid)
    return u_xx - rhs_fn(x_grid)


# ==========================================
# 6. TREINAMENTO DO SV-PINN (L-BFGS), Sec. 4.4, sem termo de contorno
#    (lambda = 0, condicao de contorno imposta de forma hard pela DAFF).
#    Registra a trajetoria passo-a-passo (loss e erro L2 relativo) via
#    jax.lax.scan sobre lbfgs.update, necessaria para o painel (b) da
#    Figura 2 (L2 relative error vs Steps).
# ==========================================
def train_sv_pinn(
        a, key,
        n_daff=64, width=512, depth=3,
        n_colloc=1024, n_test_functions=25000, tau=0.1,
        lbfgs_maxiter=5000, tol=1e-9, history_size=200,
        n_test_eval=2048
    ):
    u_exact, rhs_fn = make_problem(a)

    x_grid = grid_points(n_colloc, d=1)[:, 0]

    key, init_key, phi_key = random.split(key, 3)
    params = init_modified_mlp(init_key, n_daff=n_daff, width=width, depth=depth)
    params_flat, unflatten_fn = jax.flatten_util.ravel_pytree(params)

    phi_samples = sample_test_functions(phi_key, n=n_colloc, d=1, N=n_test_functions, tau=tau)

    x_test = jnp.linspace(0.0, 1.0, n_test_eval, dtype=jnp.float64).reshape(-1, 1)
    u_true = u_exact(x_test)
    u_true_norm = jnp.linalg.norm(u_true)

    def objective(p_flat):
        params_ = unflatten_fn(p_flat)
        R = residual_interior(params_, x_grid, rhs_fn)
        return sv_pinn_norm_squared(R, phi_samples)

    lbfgs = jaxopt.LBFGS(
        fun=objective, maxiter=lbfgs_maxiter,
        history_size=history_size, tol=tol
    )

    init_state = lbfgs.init_state(params_flat)

    def step_fn(carry, _):
        p_flat, state = carry
        p_flat, state = lbfgs.update(p_flat, state)
        params_ = unflatten_fn(p_flat)
        u_pred = modified_mlp_forward(x_test, params_)
        l2_err = jnp.linalg.norm(u_pred - u_true) / u_true_norm
        return (p_flat, state), (state.value, l2_err)

    @jax.jit
    def run_training(p_flat, state):
        return jax.lax.scan(step_fn, (p_flat, state), xs=None, length=lbfgs_maxiter)

    t0 = time.perf_counter()
    (final_p_flat, final_state), (loss_traj, err_traj) = run_training(params_flat, init_state)
    loss_traj.block_until_ready()
    train_time = time.perf_counter() - t0

    final_params = unflatten_fn(final_p_flat)
    u_pred_final = modified_mlp_forward(x_test, final_params)

    return {
        "params": final_params,
        "train_time_s": train_time,
        "loss_trajectory": loss_traj,        # shape (lbfgs_maxiter,)
        "l2_error_trajectory": err_traj,      # shape (lbfgs_maxiter,), Eq. 6.2 a cada passo
        "x_test": x_test,
        "u_true": u_true,
        "u_pred": u_pred_final,
    }


# ==========================================
# 8. LOOP PRINCIPAL: a in {1, 25, 50}, Experimento 1 (Sec. 6.1)
#    Apenas treina e salva os dados em JSON; nenhum plot e' feito aqui
#    (implementacao dos graficos fica para um outro momento).
# ==========================================
if __name__ == "__main__":
    a_values = [1, 25, 50]
    num_runs = 3            # repeticoes com sementes distintas (Sec. 6, protocolo do paper)
    tau = 0.1                # Sec. 6: tau = 0.1 para d = 1 (estabilidade numerica)
    n_daff = 64
    width = 512
    depth = 3
    n_colloc = 1024
    n_test_functions = 25000
    lbfgs_maxiter = 5000
    n_test_eval = 2048       # grid de teste, Table A.8 (1D)
    solution_run_idx = 0     # qual execucao e' salva para o plot da solucao (Fig. 2a)

    main_key = random.PRNGKey(42)

    os.makedirs("results", exist_ok=True)

    for a in a_values:
        print("\n" + "=" * 60)
        print(f"EXPERIMENTO 1 (Sec. 6.1) | a = {a} | {num_runs} execucoes | L-BFGS")
        print("=" * 60)

        runs_summary = []
        error_trajectories = []   # (num_runs, lbfgs_maxiter), para o painel (b)
        loss_trajectories = []
        solution_data = None      # dados de x, u_true, u_pred para o painel (a)

        for run in range(num_runs):
            run_key = random.fold_in(main_key, hash((a, run)) % (2 ** 31))
            out = train_sv_pinn(
                a, run_key, 
                n_daff=n_daff, width=width, depth=depth,
                n_colloc=n_colloc, n_test_functions=n_test_functions, tau=tau,
                lbfgs_maxiter=lbfgs_maxiter, n_test_eval=n_test_eval,
            )

            err_traj = np.asarray(out["l2_error_trajectory"])
            loss_traj = np.asarray(out["loss_trajectory"])
            final_err = float(err_traj[-1])
            final_loss = float(loss_traj[-1])

            print(f"  run {run + 1}/{num_runs} | loss final={final_loss:.4e} | "
                  f"L2 rel. error final={final_err:.4e} | tempo={out['train_time_s']:.1f}s")

            runs_summary.append({
                "run": run,
                "final_l2_relative_error": final_err,
                "final_loss": final_loss,
                "train_time_s": out["train_time_s"],
            })
            error_trajectories.append(err_traj)
            loss_trajectories.append(loss_traj)

            if run == solution_run_idx:
                solution_data = {
                    "x": np.asarray(out["x_test"]).flatten().tolist(),
                    "u_true": np.asarray(out["u_true"]).flatten().tolist(),
                    "u_pred": np.asarray(out["u_pred"]).flatten().tolist(),
                    "run": run,
                }

        error_trajectories = np.stack(error_trajectories, axis=0)  # (num_runs, lbfgs_maxiter)
        loss_trajectories = np.stack(loss_trajectories, axis=0)
        final_errors = np.array([r["final_l2_relative_error"] for r in runs_summary])
        print(f"\n  Erro L2 relativo final: media={final_errors.mean():.4e} "
              f"(dp={final_errors.std():.4e})")

        hyperparams = {
            "n_daff": n_daff, "width": width, "depth": depth,
            "n_colloc": n_colloc, "n_test_functions": n_test_functions,
            "lbfgs_maxiter": lbfgs_maxiter, "tau": tau, "n_test_eval": n_test_eval,
            "num_runs": num_runs, "method": "SV-PINN", "features": "DAFF", "optimizer": "L-BFGS",
        }

        # Dados para o plot da solucao SV-PINN (estilo Fig. 2a: solucao exata vs
        # aproximacao e erro pontual, u_true - u_pred).
        solution_payload = {
            "a": a,
            "hyperparameters": hyperparams,
            "solution": solution_data,
        }
        with open(f"results/exp1_poisson1d_solution_a{a}.json", "w") as f:
            json.dump(solution_payload, f, indent=4)

        # Dados para o plot do painel (b): erro L2 relativo por passo de
        # treinamento, para cada uma das `num_runs` repeticoes.
        training_curve_payload = {
            "a": a,
            "hyperparameters": hyperparams,
            "steps": list(range(1, lbfgs_maxiter + 1)),
            "l2_relative_error_per_run": error_trajectories.tolist(),
            "loss_per_run": loss_trajectories.tolist(),
            "l2_relative_error_mean": error_trajectories.mean(axis=0).tolist(),
            "l2_relative_error_std": error_trajectories.std(axis=0).tolist(),
            "runs_summary": runs_summary,
        }
        with open(f"results/exp1_poisson1d_training_curve_a{a}.json", "w") as f:
            json.dump(training_curve_payload, f, indent=4)

        print(f"  Dados salvos em results/exp1_poisson1d_solution_a{a}.json e "
              f"results/exp1_poisson1d_training_curve_a{a}.json")