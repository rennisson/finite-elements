# -*- coding: utf-8 -*-
"""
vpinn_1d_poisson.py

Implementacao de VPINNs (Variational Physics-Informed Neural Networks) para a
equacao de Poisson 1D, seguindo Kharazmi, Zhang & Karniadakis (2019),
"VPINNs: Variational Physics-Informed Neural Networks for Solving PDEs".

Problema (Secao 4/5 do artigo, eq. 4.23):

    -u''(x) = f(x),  x in (-1, 1)
    u(-1) = g,  u(1) = h

Formulacao variacional (Petrov-Galerkin): multiplicamos por uma funcao teste
v(x) com suporte compacto em (-1,1) e integramos por partes. Como v_k(+-1)=0,
o termo de contorno desaparece (Remark 4.1) e obtemos duas formas equivalentes
do residuo variacional (eq. 4.6-4.7):

    R_k^(1) = -(u_xx, v_k)_Omega           (nenhuma integracao por partes)
    R_k^(2) =  (u_x, v_k')_Omega            (uma integracao por partes)

com F_k = (f, v_k)_Omega e perda variacional (eq. 4.25):

    L_R = (1/K) sum_k (R_k - F_k)^2
    L_b = (tau/2) [ (u_NN(-1)-g)^2 + (u_NN(1)-h)^2 ]
    L   = L_R + L_b

Como no artigo (Secao 5, "Shallow to Deep VPINNs"), como a rede pode ser
profunda, as integrais nao tem forma fechada e sao aproximadas por quadratura
de Gauss-Legendre (equivalente a Gauss-Jacobi com alpha=beta=0), usando os
pontos/pesos {x_q, W_q} (eq. da Secao 5):

    R_k^(1) ~= -sum_q W_q u_xx(x_q) v_k(x_q)
    R_k^(2) ~=  sum_q W_q u_x(x_q)  v_k'(x_q)

As funcoes teste sao polinomios de Legendre deslocados (eq. 4.20):

    v_k(x) = P_{k+1}(x) - P_{k-1}(x),  k = 1, ..., K

que satisfazem v_k(-1) = v_k(1) = 0. Em vez de usar as formulas de recursao
fechadas do Apendice B (validas apenas para redes rasas com ativacao seno),
calculamos v_k e v_k' via a mesma recursao de Legendre + autodiff do JAX, o
que funciona para qualquer arquitetura de rede (rasa ou profunda) e qualquer
funcao de ativacao -- exatamente o cenario "Shallow to Deep VPINNs" da
Secao 5.

Os exemplos de solucao exata (Example 5.1) sao:
    - 'steep':          u_exact(x) = 0.1 sin(4 pi x) + tanh(5x)
    - 'boundary_layer':  u_exact(x) = 0.1 sin(4 pi x) + exp((0.01-(x+1))/0.01)

O termo de forcamento f(x) = -u_exact''(x) e obtido automaticamente via
autodiff (metodo das solucoes fabricadas), sem necessidade de deriva-lo a mao.

Convencoes de codigo e formato dos arquivos de saida (JSON) seguem o script
de referencia `1d_poisson.py`:
    - dados_vpinn_1d_<tag>.json          -> resumo (erro L2, tempos, etc.)
    - pontos_vpinn_1d_<tag>.json         -> predicao final da rede + pesos
    - curva_treino_vpinn_1d_<tag>.json   -> trajetoria de loss / erro L2
"""
import jax
import jax.flatten_util
import jax.numpy as jnp
import json
import numpy as np
import os
import time

from jax import random, config

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

activation_function = jax.nn.tanh


K_TEST_FUNCTIONS = 60
Q_QUADRATURE = 100
NUM_STEPS = 50000
EVAL_FREQ = 50
NUM_RUNS = 5
HIDDEN_LAYER_CONFIGS = [1, 2]

NEURONS_PER_LAYER = 20   # N=20 neuronios por camada, como na Tabela do Ex. 5.1
LEARNING_RATE = 1e-3     # Adam, lr = 1e-3 (Tabela 1/2 do artigo)

# tau (parametro de penalidade de contorno) por caso, como nas legendas das
# Fig. 9-10 do artigo.
TAU_VPINN = {"steep": 25.0, "boundary_layer": 10.0}

main_key = random.PRNGKey(42)


# ==========================================
# 1. SOLUCOES EXATAS (METODO DAS SOLUCOES FABRICADAS, Example 5.1)
# ==========================================
def u_exact_steep(x):
    """u_exact(x) = 0.1 sin(4 pi x) + tanh(5x)  (eq. 5.1)"""
    return 0.1 * jnp.sin(4 * jnp.pi * x) + jnp.tanh(5.0 * x)


def u_exact_boundary_layer(x):
    """u_exact(x) = 0.1 sin(4 pi x) + exp((0.01-(x+1))/0.01)  (eq. 5.2)"""
    return 0.1 * jnp.sin(4 * jnp.pi * x) + jnp.exp((0.01 - (x + 1.0)) / 0.01)


EXACT_SOLUTIONS = {
    "steep": u_exact_steep,
    "boundary_layer": u_exact_boundary_layer,
}


def make_force_term(u_exact_fn):
    """f(x) = -u_exact''(x), via diferenciacao automatica (2x jax.grad)."""
    du_dx = jax.grad(u_exact_fn)
    d2u_dx2 = jax.grad(du_dx)

    def f_fn(x):
        return -d2u_dx2(x)

    return f_fn


# ==========================================
# 2. FUNCOES TESTE DE LEGENDRE (eq. 4.20) E SUAS DERIVADAS
# ==========================================
def legendre_stack(x, kmax):
    """
    Retorna [P_0(x), P_1(x), ..., P_kmax(x)] via a recursao de Legendre
    (eq. B.1 do Apendice B): P_k = ((2k-1) x P_{k-1} - (k-1) P_{k-2}) / k.

    x: escalar. kmax: inteiro estatico (grau maximo).
    """
    P0 = jnp.ones_like(x)
    P1 = x
    Ps = [P0, P1]
    for k in range(2, kmax + 1):
        Pk = ((2 * k - 1) * x * Ps[-1] - (k - 1) * Ps[-2]) / k
        Ps.append(Pk)
    return jnp.stack(Ps)  # shape (kmax+1,)


def v_vector(x, K):
    """
    v_k(x) = P_{k+1}(x) - P_{k-1}(x), k = 1, ..., K  (eq. 4.20).
    Retorna vetor de shape (K,). Escalar x.
    """
    P = legendre_stack(x, K + 1)          # graus 0..K+1, shape (K+2,)
    return P[2:] - P[:-2]                  # v_k = P_{k+1} - P_{k-1}


def make_test_function_tables(x_q, K):
    """
    Avalia v_k(x_q) e v_k'(x_q) para todos os k=1..K e todos os pontos de
    quadratura x_q, usando autodiff (jax.jacfwd) sobre a recursao de
    Legendre -- generaliza o Apendice B para qualquer K sem precisar
    programar as formulas de recursao fechadas.

    Retorna V, Vx com shape (Q, K).
    """
    def v_and_vx(x_scalar):
        v = v_vector(x_scalar, K)
        vx = jax.jacfwd(lambda xx: v_vector(xx, K))(x_scalar)
        return v, vx

    V, Vx = jax.vmap(v_and_vx)(x_q)
    return V, Vx


# ==========================================
# 3. QUADRATURA DE GAUSS-LEGENDRE EM (-1, 1)
# ==========================================
def gauss_legendre_quadrature(Q, dtype=jnp.float64):
    """Pontos e pesos de Gauss-Legendre em (-1,1) (equivalente a
    Gauss-Jacobi com alpha=beta=0, usado no artigo)."""
    x_q, w_q = np.polynomial.legendre.leggauss(Q)
    return jnp.asarray(x_q, dtype=dtype), jnp.asarray(w_q, dtype=dtype)


# ==========================================
# 4. REDE NEURAL (MLP) E DERIVADAS VIA AUTODIFF
#    (mesma convencao de parametros/forward do 1d_poisson.py)
# ==========================================
def init_mlp_params(key, widths, dtype=jnp.float64):
    """widths = [n_in, n_hidden_1, ..., n_hidden_L, n_out]."""
    initializer = jax.nn.initializers.glorot_normal()
    keys = random.split(key, len(widths) - 1)
    params = []
    for k, lin, lout in zip(keys, widths[:-1], widths[1:]):
        W = initializer(k, (lin, lout), dtype)
        B = initializer(k, (1, lout), dtype)
        params.append({"W": W, "B": B})
    return params


@jax.jit
def forward(x, params):
    """x: shape (batch, 1). Retorna shape (batch, 1)."""
    *hidden, output = params
    for layer in hidden:
        x = activation_function(x @ layer["W"] + layer["B"])
    return x @ output["W"] + output["B"]


def u_scalar(x, params):
    """u_NN avaliado em um ponto escalar x."""
    return forward(x.reshape(1, 1), params)[0, 0]


def u_x_scalar(x, params):
    return jax.grad(u_scalar, argnums=0)(x, params)


def u_xx_scalar(x, params):
    return jax.grad(u_x_scalar, argnums=0)(x, params)


# ==========================================
# 5. RESIDUOS/PERDAS
# ==========================================
def compute_F(x_q, w_q, V, f_fn):
    """F_k = sum_q W_q f(x_q) v_k(x_q)  (nao depende dos parametros da rede,
    calculado uma unica vez por problema)."""
    f_q = jax.vmap(f_fn)(x_q)
    return jnp.einsum("q,q,qk->k", w_q, f_q, V)


def vpinn_loss_R1(params, x_q, w_q, V, F, tau, g, h):
    """L^(1): R_k^(1) = -sum_q W_q u_xx(x_q) v_k(x_q)  (eq. 4.6, sem
    integracao por partes)."""
    uxx_q = jax.vmap(u_xx_scalar, in_axes=(0, None))(x_q, params)
    R = -jnp.einsum("q,q,qk->k", w_q, uxx_q, V)
    L_R = jnp.mean((R - F) ** 2)
    u_m1 = u_scalar(jnp.asarray(-1.0), params)
    u_p1 = u_scalar(jnp.asarray(1.0), params)
    L_b = 0.5 * ((u_m1 - g) ** 2 + (u_p1 - h) ** 2)
    return L_R + tau * L_b


def vpinn_loss_R2(params, x_q, w_q, Vx, F, tau, g, h):
    """L^(2): R_k^(2) = sum_q W_q u_x(x_q) v_k'(x_q)  (eq. 4.7, uma
    integracao por partes -- forma fraca/simetrica, so precisa de u_x)."""
    ux_q = jax.vmap(u_x_scalar, in_axes=(0, None))(x_q, params)
    R = jnp.einsum("q,q,qk->k", w_q, ux_q, Vx)
    L_R = jnp.mean((R - F) ** 2)
    u_m1 = u_scalar(jnp.asarray(-1.0), params)
    u_p1 = u_scalar(jnp.asarray(1.0), params)
    L_b = 0.5 * ((u_m1 - g) ** 2 + (u_p1 - h) ** 2)
    return L_R + tau * L_b


# ==========================================
# 6. OTIMIZADOR ADAM (implementacao manual, sem dependencia extra)
# ==========================================
def init_adam_state(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": m, "v": v, "t": jnp.asarray(0, dtype=jnp.int32)}


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * (g ** 2), state["v"], grads)
    t_f = t.astype(jnp.float64)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1 ** t_f), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2 ** t_f), v)
    updates = jax.tree_util.tree_map(lambda mh, vh: lr * mh / (jnp.sqrt(vh) + eps), m_hat, v_hat)
    new_params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
    return new_params, {"m": m, "v": v, "t": t}


def train_adam(params, loss_fn, num_steps, eval_freq, lr, X_test, Y_test):
    """
    Treina `params` minimizando `loss_fn(params)` via Adam, registrando a
    loss e o erro L2 relativo (contra X_test/Y_test) a cada `eval_freq`
    passos. Estrutura em dois niveis de jax.lax.scan (bloco de eval_freq
    passos de otimizacao, depois uma avaliacao), igual ao padrao usado em
    `1d_poisson.py` para as trajetorias de treino do L-BFGS.
    """
    opt_state = init_adam_state(params)
    num_blocks = num_steps // eval_freq

    def opt_step(carry, _):
        p, s = carry
        loss, grads = jax.value_and_grad(loss_fn)(p)
        p, s = adam_update(p, grads, s, lr)
        return (p, s), loss

    def block_fn(carry, _):
        carry, losses = jax.lax.scan(opt_step, carry, xs=None, length=eval_freq)
        p, s = carry
        Y_nn = forward(X_test, p)
        l2_err = jnp.linalg.norm(Y_nn - Y_test) / jnp.linalg.norm(Y_test)
        return carry, (losses[-1], l2_err)

    run_fn = jax.jit(lambda p, s: jax.lax.scan(block_fn, (p, s), xs=None, length=num_blocks))
    (params_final, _), (loss_traj, err_traj) = run_fn(params, opt_state)
    return params_final, np.asarray(loss_traj), np.asarray(err_traj)


# ==========================================
# 7. LOOP PRINCIPAL DE EXPERIMENTOS
# ==========================================
X_TEST = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64).reshape(-1, 1)

for solution_name, u_exact_fn in EXACT_SOLUTIONS.items():
    f_fn = make_force_term(u_exact_fn)
    g = float(u_exact_fn(jnp.asarray(-1.0)))
    h = float(u_exact_fn(jnp.asarray(1.0)))
    Y_TEST = jax.vmap(u_exact_fn)(X_TEST[:, 0]).reshape(-1, 1)

    # ------ dados fixos da formulacao variacional (nao dependem da rede) ------
    x_q, w_q = gauss_legendre_quadrature(Q_QUADRATURE)
    V, Vx = make_test_function_tables(x_q, K_TEST_FUNCTIONS)
    F = compute_F(x_q, w_q, V, f_fn)

    for L in HIDDEN_LAYER_CONFIGS:
        width = [1] + [NEURONS_PER_LAYER] * L + [1]

        methods = {
            "vpinnR1": lambda p: vpinn_loss_R1(p, x_q, w_q, V, F, TAU_VPINN[solution_name], g, h),
            "vpinnR2": lambda p: vpinn_loss_R2(p, x_q, w_q, Vx, F, TAU_VPINN[solution_name], g, h),
        }

        for method_tag, loss_builder in methods.items():
            print("\n" + "=" * 20)
            print(f"SOLUCAO={solution_name} | METODO={method_tag} | L={L} hidden layers")
            print("=" * 20)

            loss_trajectories, err_trajectories, l2_errors = [], [], []
            acc_train_time, acc_eval_time = 0.0, 0.0
            Y_nn_final, params_final_flat = None, None

            for run in range(NUM_RUNS):
                run_key = random.fold_in(main_key, hash((solution_name, method_tag, L, run)) % (2**31))
                params = init_mlp_params(run_key, width)

                t0 = time.perf_counter()
                params, loss_traj, err_traj = train_adam(
                    params, loss_builder, NUM_STEPS, EVAL_FREQ, LEARNING_RATE, X_TEST, Y_TEST
                )
                jax.block_until_ready(params)
                t_train = time.perf_counter() - t0
                acc_train_time += t_train

                loss_trajectories.append(loss_traj)
                err_trajectories.append(err_traj)

                t0 = time.perf_counter()
                Y_nn = forward(X_TEST, params)
                Y_nn.block_until_ready()
                acc_eval_time += time.perf_counter() - t0

                rel_l2 = float(jnp.linalg.norm(Y_nn - Y_TEST) / jnp.linalg.norm(Y_TEST))
                l2_errors.append(rel_l2)
                print(f"Run {run + 1}/{NUM_RUNS}: erro L2 relativo = {rel_l2:.6e} "
                      f"(treino: {t_train:.2f}s)")

                if run == NUM_RUNS - 1:
                    Y_nn_final = np.asarray(Y_nn)
                    flat_params, _ = jax.flatten_util.ravel_pytree(params)
                    params_final_flat = np.asarray(flat_params)

            avg_rel_l2 = float(np.mean(l2_errors))
            median_rel_l2 = float(np.median(l2_errors))
            avg_train_time = acc_train_time / NUM_RUNS
            avg_eval_time = acc_eval_time / NUM_RUNS

            print(f"--- Erro L2 relativo medio: {avg_rel_l2:.6e} | "
                  f"mediana: {median_rel_l2:.6e} ---")

            tag = f"{solution_name}_{method_tag}_L{L}"

            results = {
                "architecture": width,
                "num_hidden_layers": L,
                "method": f"VPINN {method_tag[-2:]} - Legendre",
                "exact_solution": solution_name,
                "n_test_functions": K_TEST_FUNCTIONS,
                "n_quadrature_points": Q_QUADRATURE,
                "tau": TAU_VPINN[solution_name],
                "learning_rate": LEARNING_RATE,
                "num_iterations": NUM_STEPS,
                "num_runs_avg": NUM_RUNS,
                "error_relativo_medio": avg_rel_l2,
                "error_relativo_mediana": median_rel_l2,
                "time_training": avg_train_time,
                "time_evaluation": avg_eval_time,
                "num_params": int(params_final_flat.shape[0]),
            }
            with open(f"dados_vpinn_1d_{tag}.json", "w") as fjson:
                json.dump(results, fjson, indent=4)

            points = {
                "x": X_TEST.flatten().tolist(),
                "y_exact": np.asarray(Y_TEST).flatten().tolist(),
                "y_nn": Y_nn_final.flatten().tolist(),
                "network_weights": params_final_flat.tolist(),
            }
            with open(f"pontos_vpinn_1d_{tag}.json", "w") as fjson:
                json.dump(points, fjson, indent=4)

            loss_trajectories = np.stack(loss_trajectories, axis=0)
            err_trajectories = np.stack(err_trajectories, axis=0)
            training_curve = {
                "architecture": width,
                "method": f"VPINN {method_tag[-2:]} - Legendre",
                "exact_solution": solution_name,
                "num_iterations": NUM_STEPS,
                "num_runs": NUM_RUNS,
                "steps": list(range(EVAL_FREQ, NUM_STEPS + 1, EVAL_FREQ)),
                "l2_relative_error_per_run": err_trajectories.tolist(),
                "loss_per_run": loss_trajectories.tolist(),
                "l2_relative_error_mean": err_trajectories.mean(axis=0).tolist(),
                "l2_relative_error_std": err_trajectories.std(axis=0).tolist(),
            }
            with open(f"curva_treino_vpinn_1d_{tag}.json", "w") as fjson:
                json.dump(training_curve, fjson, indent=4)

            print(f"Dados salvos para tag={tag}")

print("\nConcluido. Todos os arquivos JSON foram salvos no diretorio atual.")