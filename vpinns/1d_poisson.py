# -*- coding: utf-8 -*-
"""
vpinn_1d_poisson_unit_interval.py

VPINN (Variational Physics-Informed Neural Network) para o problema de
Poisson 1D do Cap. 4.1 do TCC (analogo ao Example 5.1 de Kharazmi, Zhang
& Karniadakis 2019, mas em (0,1) em vez de (-1,1)):

    u''(x) = (4x^3 - 6x) exp(-x^2),  x in (0, 1)
    u(0) = 0
    u(1) = exp(-1)

Solucao analitica: u_exact(x) = x exp(-x^2).

Diferencas em relacao a vpinn_1d_poisson.py (Example 5.1, dominio (-1,1)):
    - o dominio fisico e (0,1), nao (-1,1);
    - a equacao e u''(x) = f(x) (sem o sinal negativo de -u''=f), o que
      inverte o sinal dos residuos variacionais R^(1) e R^(2) em relacao
      ao script de referencia.

As funcoes teste de Legendre (eq. 4.20 do artigo) sao definidas no
dominio de referencia xi in (-1,1), onde vivem naturalmente
(v_k(+-1)=0), e mapeadas para o dominio fisico x in (0,1) via a
transformacao afim

    x = (xi + 1) / 2,  dx = (1/2) dxi,  dxi/dx = 2.

Formulacao variacional (multiplica-se a EDO por v_k e integra-se em
(0,1)); como v_k(0) = v_k(1) = 0, o termo de contorno da integracao por
partes desaparece:

    R_k^(1) =  (u_xx, v_k)_(0,1)            (sem IBP)
    R_k^(2) = -(u_x,  v_k')_(0,1)           (uma IBP)
    F_k     =  (f, v_k)_(0,1)

    L_R = (1/K) sum_k (R_k - F_k)^2
    L_b = (1/2) [ (u_NN(0)-g)^2 + (u_NN(1)-h)^2 ]
    L   = L_R + tau * L_b

Integrais aproximadas por quadratura de Gauss-Legendre em (-1,1),
mapeada para (0,1). Ativacao da rede: tanh (Secao 5, "Shallow to Deep
VPINNs" -- redes profundas exigem quadratura numerica, sem forma
fechada).

O erro L2 relativo (avaliacao e curva de treino) e calculado contra o
ground truth gerado por `ground_truth_poisson_1d.py` (arquivo
GT_JSON_PATH, campos "X" e "U_true"), nao contra u_exact avaliada
diretamente em um novo grid. `u_exact_fn` continua sendo usada apenas
para obter f(x) por autodiff (metodo das solucoes fabricadas) e os
valores de contorno g, h.

Saidas (mesmo padrao de vpinn_1d_poisson.py):
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

from pathlib import Path

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

activation_function = jax.nn.tanh

K_TEST_FUNCTIONS = 60
Q_QUADRATURE = 100
NUM_STEPS = 50000
EVAL_FREQ = 50
NUM_RUNS = 10
HIDDEN_LAYER_CONFIGS = [1, 2, 3]

NEURONS_PER_LAYER = [5, 10, 20, 40]
LEARNING_RATE = 1e-3
TAU_VPINN = 10.0

X_LEFT, X_RIGHT = 0.0, 1.0

GT_JSON_PATH = "gt_poisson_1d.json"  # gerado por ground_truth_poisson_1d.py

main_key = random.PRNGKey(42)


# ==========================================
# 1. SOLUCAO EXATA E TERMO DE FORCAMENTO (eq. 5 do TCC)
# ==========================================
def u_exact_fn(x):
    """u_exact(x) = x exp(-x^2)."""
    return x * jnp.exp(-x ** 2)


def make_force_term(u_exact_fn):
    """f(x) = u_exact''(x), via diferenciacao automatica (2x jax.grad).
    Observe o sinal: a EDO e u''(x) = f(x), sem o sinal negativo de
    -u''=f usado no Example 5.1 do artigo."""
    du_dx = jax.grad(u_exact_fn)
    d2u_dx2 = jax.grad(du_dx)

    def f_fn(x):
        return d2u_dx2(x)

    return f_fn


# ==========================================
# 2. FUNCOES TESTE DE LEGENDRE (eq. 4.20) NO DOMINIO DE REFERENCIA
# ==========================================
def legendre_stack(xi, kmax):
    """[P_0(xi), ..., P_kmax(xi)] via recursao de Legendre (eq. B.1)."""
    P0 = jnp.ones_like(xi)
    P1 = xi
    Ps = [P0, P1]
    for k in range(2, kmax + 1):
        Pk = ((2 * k - 1) * xi * Ps[-1] - (k - 1) * Ps[-2]) / k
        Ps.append(Pk)
    return jnp.stack(Ps)


def v_vector(xi, K):
    """v_k(xi) = P_{k+1}(xi) - P_{k-1}(xi), k=1..K (eq. 4.20), definidas
    no dominio de referencia xi in (-1,1) -> v_k(+-1) = 0."""
    P = legendre_stack(xi, K + 1)
    return P[2:] - P[:-2]


def make_test_function_tables(xi_q, K):
    """
    Avalia v_k(xi_q) e a derivada fisica dv_k/dx(xi_q) = dv_k/dxi * 2
    (regra da cadeia de x = (xi+1)/2) para todos os k=1..K e todos os
    pontos de quadratura de referencia xi_q, via autodiff (jax.jacfwd).

    Retorna V, Vx_phys com shape (Q, K).
    """
    def v_and_vx(xi_scalar):
        v = v_vector(xi_scalar, K)
        dv_dxi = jax.jacfwd(lambda xx: v_vector(xx, K))(xi_scalar)
        return v, dv_dxi

    V, dV_dxi = jax.vmap(v_and_vx)(xi_q)
    Vx_phys = 2.0 * dV_dxi  # dxi/dx = 2 no mapeamento x = (xi+1)/2
    return V, Vx_phys


# ==========================================
# 3. QUADRATURA DE GAUSS-LEGENDRE MAPEADA PARA (0, 1)
# ==========================================
def gauss_legendre_quadrature_unit_interval(Q, dtype=jnp.float64):
    """
    Pontos/pesos de Gauss-Legendre em (-1,1), mapeados para o dominio
    fisico (0,1) via x = (xi+1)/2, dx = (1/2) dxi.

    Retorna:
        xi_q      -- pontos no dominio de referencia (para as funcoes teste)
        x_q       -- pontos fisicos correspondentes (para a rede)
        w_q_phys  -- pesos ja escalados pelo jacobiano (1/2)
    """
    xi_q, w_q = np.polynomial.legendre.leggauss(Q)
    xi_q = jnp.asarray(xi_q, dtype=dtype)
    w_q = jnp.asarray(w_q, dtype=dtype)
    x_q = 0.5 * (xi_q + 1.0)
    w_q_phys = 0.5 * w_q
    return xi_q, x_q, w_q_phys


# ==========================================
# 4. REDE NEURAL (MLP) E DERIVADAS VIA AUTODIFF
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
def compute_F(x_q, w_q_phys, V, f_fn):
    """F_k = sum_q W_q f(x_q) v_k(x_q) (nao depende dos parametros da
    rede, calculado uma unica vez)."""
    f_q = jax.vmap(f_fn)(x_q)
    return jnp.einsum("q,q,qk->k", w_q_phys, f_q, V)


def vpinn_loss_R1(params, x_q, w_q_phys, V, F, tau, g, h):
    """L^(1): R_k^(1) = sum_q W_q u_xx(x_q) v_k(x_q) (sem IBP). Sinal
    positivo pois a EDO e u''=f (nao -u''=f, como no Example 5.1)."""
    uxx_q = jax.vmap(u_xx_scalar, in_axes=(0, None))(x_q, params)
    R = jnp.einsum("q,q,qk->k", w_q_phys, uxx_q, V)
    L_R = jnp.mean((R - F) ** 2)
    u_left = u_scalar(jnp.asarray(X_LEFT), params)
    u_right = u_scalar(jnp.asarray(X_RIGHT), params)
    L_b = 0.5 * ((u_left - g) ** 2 + (u_right - h) ** 2)
    return L_R + tau * L_b


def vpinn_loss_R2(params, x_q, w_q_phys, Vx_phys, F, tau, g, h):
    """L^(2): R_k^(2) = -sum_q W_q u_x(x_q) v_k'(x_q) (uma IBP; o termo
    de contorno se anula pois v_k(0)=v_k(1)=0)."""
    ux_q = jax.vmap(u_x_scalar, in_axes=(0, None))(x_q, params)
    R = -jnp.einsum("q,q,qk->k", w_q_phys, ux_q, Vx_phys)
    L_R = jnp.mean((R - F) ** 2)
    u_left = u_scalar(jnp.asarray(X_LEFT), params)
    u_right = u_scalar(jnp.asarray(X_RIGHT), params)
    L_b = 0.5 * ((u_left - g) ** 2 + (u_right - h) ** 2)
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
    passos. Dois niveis de jax.lax.scan (bloco de eval_freq passos de
    otimizacao, depois uma avaliacao), igual ao padrao usado em
    `vpinn_1d_poisson.py`.
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
# 7. GROUND TRUTH (ground_truth_poisson_1d.py) E LOOP PRINCIPAL
# ==========================================
def load_ground_truth(path):
    """Carrega X e U_true gerados por ground_truth_poisson_1d.py e usados
    como referencia para o erro L2 relativo (avaliacao e curva de treino)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ground truth '{path}' nao encontrado. Rode "
            f"ground_truth_poisson_1d.py antes deste script."
        )
    with open(path, "r", encoding="utf-8") as f_gt:
        gt_data = json.load(f_gt)
    X = jnp.asarray(gt_data["X"], dtype=jnp.float64)
    U_true = jnp.asarray(gt_data["U_true"], dtype=jnp.float64)
    return X, U_true


X_TEST, Y_TEST = load_ground_truth(GT_JSON_PATH)

f_fn = make_force_term(u_exact_fn)
g = float(u_exact_fn(jnp.asarray(X_LEFT)))
h = float(u_exact_fn(jnp.asarray(X_RIGHT)))

# ------ dados fixos da formulacao variacional (nao dependem da rede) ------
xi_q, x_q, w_q_phys = gauss_legendre_quadrature_unit_interval(Q_QUADRATURE)
V, Vx_phys = make_test_function_tables(xi_q, K_TEST_FUNCTIONS)
F = compute_F(x_q, w_q_phys, V, f_fn)

for L in HIDDEN_LAYER_CONFIGS:
    for NEURONS in NEURONS_PER_LAYER:
        width = [1] + [NEURONS] * L + [1]

        methods = {
            "R1": lambda p: vpinn_loss_R1(p, x_q, w_q_phys, V, F, TAU_VPINN, g, h),
            "R2": lambda p: vpinn_loss_R2(p, x_q, w_q_phys, Vx_phys, F, TAU_VPINN, g, h),
        }

        for method_tag, loss_builder in methods.items():
            print("\n" + "=" * 20)
            print(f"PROBLEMA=poisson_1d | METODO={method_tag} | ARQ={width}")
            print("=" * 20)

            loss_trajectories, err_trajectories, l2_errors = [], [], []
            acc_train_time, acc_eval_time = 0.0, 0.0
            Y_nn_final, params_final_flat = None, None

            for run in range(NUM_RUNS):
                run_key = random.fold_in(main_key, hash((method_tag, L, run)) % (2 ** 31))
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

            arch = [NEURONS] * L + [1]
            arch_str = "_".join(map(str, width))
            tag = f"poisson_1d_{method_tag}_{arch_str}"

            output_dir = Path("vpinn_poisson_1d")
            output_dir.mkdir(exist_ok=True)

            nome_arquivo = output_dir / f"dados_vpinn_{tag}.json"
            results = {
                "architecture": width,
                "num_hidden_layers": L,
                "method": f"VPINN {method_tag[-2:]} - Legendre",
                "problem": "poisson_1d",
                "domain": [X_LEFT, X_RIGHT],
                "n_test_functions": K_TEST_FUNCTIONS,
                "n_quadrature_points": Q_QUADRATURE,
                "tau": TAU_VPINN,
                "learning_rate": LEARNING_RATE,
                "num_iterations": NUM_STEPS,
                "num_runs_avg": NUM_RUNS,
                "error_relativo_medio": avg_rel_l2,
                "error_relativo_mediana": median_rel_l2,
                "time_training": avg_train_time,
                "time_evaluation": avg_eval_time,
                "num_params": int(params_final_flat.shape[0]),
            }
            with open(nome_arquivo, "w") as fjson:
                json.dump(results, fjson, indent=4)

            nome_arquivo = output_dir / f"pontos_vpinn_{tag}.json"
            points = {
                "x": X_TEST.flatten().tolist(),
                "y_exact": np.asarray(Y_TEST).flatten().tolist(),
                "y_nn": Y_nn_final.flatten().tolist(),
                "network_weights": params_final_flat.tolist(),
            }
            with open(nome_arquivo, "w") as fjson:
                json.dump(points, fjson, indent=4)

            nome_arquivo = output_dir / f"curva_treino_vpinn_{tag}.json"
            loss_trajectories = np.stack(loss_trajectories, axis=0)
            err_trajectories = np.stack(err_trajectories, axis=0)
            training_curve = {
                "architecture": width,
                "method": f"VPINN {method_tag[-2:]} - Legendre",
                "problem": "poisson_1d",
                "num_iterations": NUM_STEPS,
                "num_runs": NUM_RUNS,
                "steps": list(range(EVAL_FREQ, NUM_STEPS + 1, EVAL_FREQ)),
                "l2_relative_error_per_run": err_trajectories.tolist(),
                "loss_per_run": loss_trajectories.tolist(),
                "l2_relative_error_mean": err_trajectories.mean(axis=0).tolist(),
                "l2_relative_error_std": err_trajectories.std(axis=0).tolist(),
            }
            with open(nome_arquivo, "w") as fjson:
                json.dump(training_curve, fjson, indent=4)

            print(f"Dados salvos para tag={tag}")

print("\nConcluido. Todos os arquivos JSON foram salvos no diretorio atual.")
