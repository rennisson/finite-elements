# -*- coding: utf-8 -*-
"""
vpinn_poisson_2d.py

VPINN (Variational Physics-Informed Neural Network) para o problema de
Poisson 2D (mesmo problema de pinn_poisson_2d.py / 2d_poisson.py):

    Delta u(x, y) = 2( x^4(3y-2) + x^3(4-6y) + x^2(6y^3-12y^2+9y-2)
                        - 6x(y-1)^2 y + (y-1)^2 y ),   (x, y) in (0,1)^2
    d_n u(0, y) = 0,  y in [0, 1]
    d_n u(1, y) = 0,  y in [0, 1]
    u(x, 0)     = 0,  x in [0, 1]
    d_n u(x, 1) = 0,  x in [0, 1]

Solucao analitica: u_exact(x, y) = x^2 (x-1)^2 y (y-1)^2.

Extensao direta de vpinn_1d_poisson.py para 2D, mantendo o mesmo padrao:
    - funcoes teste de Legendre v_k(xi) = P_{k+1}(xi) - P_{k-1}(xi)
      (eq. 4.20), que se anulam em xi=+-1; em 2D usamos o produto
      tensorial v_{kl}(x,y) = v_k(x) v_l(y), que se anula em TODA a
      fronteira do quadrado (0,1)^2;
    - como as unicas funcoes teste usadas se anulam em toda a fronteira,
      o termo de fronteira da integracao por partes desaparece por
      completo (tanto no lado de Neumann quanto no de Dirichlet) -- a
      informacao das 4 condicoes de contorno (Eq. 7 do TCC) e' reinjetada
      via a penalidade L_b (mesmo papel que o L_b do caso 1D, so' que
      agora com 4 termos, um por aresta, cada um com N_G_BOUNDARY pontos
      de colocacao amostrados por LHS e MANTIDOS FIXOS durante toda a
      otimizacao L-BFGS de cada run -- mesmo padrao usado para os pontos
      de contorno de svpinn_1d_poisson.py / 2d_poisson.py, diferente do
      resampling a cada passo usado no treino por Adam em
      pinn_poisson_2d.py);
    - quadratura de Gauss-Legendre tensorial (produto de duas quadraturas
      1D identicas as usadas no caso 1D);
    - otimizacao por L-BFGS (jaxopt.LBFGS) no mesmo padrao em dois niveis
      (fori_loop de EVAL_FREQ passos dentro de um lax.scan de
      NUM_STEPS // EVAL_FREQ blocos) usado em vpinn_1d_poisson.py e
      svpinn_1d_poisson.py.

f(x,y) = Delta u_exact(x,y) e os alvos de contorno (g em y=0, alvos de
Neumann em x=0, x=1, y=1) sao obtidos por autodiff de u_exact_fn (metodo
das solucoes fabricadas), do mesmo jeito que f(x)=u_exact''(x) e g,h sao
obtidos no caso 1D.

O erro L2 relativo (avaliacao e curva de treino) e calculado contra o
ground truth gerado por `ground_truth_poisson_2d.py` (arquivo
GT_JSON_PATH, campos "X", "Y", "U_true").

Saidas (mesmo padrao de vpinn_1d_poisson.py):
    - dados_vpinn_<tag>.json          -> resumo (erro L2, tempos, etc.)
    - pontos_vpinn_<tag>.json         -> predicao final da rede + pesos
    - curva_treino_vpinn_<tag>.json   -> trajetoria de loss / erro L2
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
from scipy.stats import qmc

from pathlib import Path

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

activation_function = jax.nn.tanh

# Por eixo; produto tensorial => K_TEST_FUNCTIONS**2 funcoes teste e
# Q_QUADRATURE**2 pontos de quadratura no total (contra 60 e 100 no 1D).
# Reduzido em relacao ao 1D para manter o custo das tabelas 2D (Q^2 x K^2)
# e das avaliacoes de Laplaciano/gradiente por autodiff em Q^2 pontos
# tratavel; ajuste se quiser mais resolucao.
K_TEST_FUNCTIONS = 20
Q_QUADRATURE = 40
N_G_BOUNDARY = 250          # pontos de colocacao por aresta (mesmo N_g de pinn_poisson_2d.py)
NUM_STEPS = 35000
EVAL_FREQ = 50
NUM_RUNS = 10
HIDDEN_LAYER_CONFIGS = [2, 3, 4, 5]

NEURONS_PER_LAYER = [20, 60]
# Nao usada para treinar (L-BFGS nao tem taxa de aprendizagem fixa);
# mantida apenas para preservar a estrutura do JSON de saida ("dados_...").
LEARNING_RATE = 1e-3
TAU_VPINN = 10.0

LBFGS_HISTORY_SIZE = 100
LBFGS_TOL = 1e-12

X_LEFT, X_RIGHT = 0.0, 1.0
Y_BOTTOM, Y_TOP = 0.0, 1.0

GT_JSON_PATH = "gt_poisson_2d.json"  # gerado por ground_truth_poisson_2d.py

main_key = random.PRNGKey(42)


# ==========================================
# 1. SOLUCAO EXATA E TERMO DE FORCAMENTO (Eq. 6/7 do TCC)
# ==========================================
def u_exact_fn(x, y):
    """u_exact(x, y) = x^2 (x-1)^2 y (y-1)^2."""
    return x ** 2 * (x - 1) ** 2 * y * (y - 1) ** 2


def make_force_term_2d(u_exact_fn):
    """f(x,y) = Delta u_exact(x,y), via diferenciacao automatica.
    Retorna tambem du_exact/dx e du_exact/dy, usados como alvo das
    condicoes de Neumann (Eq. 7)."""
    du_dx_exact = jax.grad(u_exact_fn, argnums=0)
    du_dy_exact = jax.grad(u_exact_fn, argnums=1)

    def f_fn(x, y):
        u_xx = jax.grad(du_dx_exact, argnums=0)(x, y)
        u_yy = jax.grad(du_dy_exact, argnums=1)(x, y)
        return u_xx + u_yy

    return f_fn, du_dx_exact, du_dy_exact


# ==========================================
# 2. FUNCOES TESTE DE LEGENDRE (eq. 4.20), PRODUTO TENSORIAL 2D
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

    Retorna V, Vx_phys com shape (Q, K). Reutilizada identica para os
    eixos x e y (mesma quadratura/funcoes teste 1D nos dois eixos).
    """
    def v_and_vx(xi_scalar):
        v = v_vector(xi_scalar, K)
        dv_dxi = jax.jacfwd(lambda xx: v_vector(xx, K))(xi_scalar)
        return v, dv_dxi

    V, dV_dxi = jax.vmap(v_and_vx)(xi_q)
    Vx_phys = 2.0 * dV_dxi  # dxi/dx = 2 no mapeamento x = (xi+1)/2
    return V, Vx_phys


def build_2d_test_function_tables(V, Vx_phys, Q, K):
    """
    Produto tensorial v_{kl}(x,y) = v_k(x) v_l(y) e suas derivadas
    fisicas, achatados para (Q*Q, K*K) -- eixo 0 indexa os pontos de
    quadratura (q,r) -> q*Q+r, eixo 1 indexa os pares de teste (k,l) ->
    k*K+l, na mesma ordem (C-order) usada para achatar o grid de
    quadratura fisico em xy_quad_flat.
    """
    V2D = jnp.einsum("qk,rl->qrkl", V, V)
    Vx2D = jnp.einsum("qk,rl->qrkl", Vx_phys, V)   # d/dx (so' o fator em x)
    Vy2D = jnp.einsum("qk,rl->qrkl", V, Vx_phys)   # d/dy (so' o fator em y)
    QQ, KK = Q * Q, K * K
    return (V2D.reshape(QQ, KK), Vx2D.reshape(QQ, KK), Vy2D.reshape(QQ, KK))


# ==========================================
# 3. QUADRATURA DE GAUSS-LEGENDRE MAPEADA PARA (0, 1), TENSORIAL 2D
# ==========================================
def gauss_legendre_quadrature_unit_interval(Q, dtype=jnp.float64):
    """Pontos/pesos de Gauss-Legendre em (-1,1), mapeados para (0,1) via
    x = (xi+1)/2, dx = (1/2) dxi. Retorna xi_q, x_q, w_q_phys (shape (Q,))."""
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
def forward(xy, params):
    """xy: shape (batch, 2). Retorna shape (batch, 1)."""
    x = xy
    *hidden, output = params
    for layer in hidden:
        x = activation_function(x @ layer["W"] + layer["B"])
    return x @ output["W"] + output["B"]


def u_scalar(x, y, params):
    """u_NN avaliado em um ponto escalar (x, y)."""
    return forward(jnp.array([[x, y]]), params)[0, 0]


def u_x_scalar(x, y, params):
    return jax.grad(lambda x_: u_scalar(x_, y, params), argnums=0)(x)


def u_y_scalar(x, y, params):
    return jax.grad(lambda y_: u_scalar(x, y_, params), argnums=0)(y)


def laplacian_scalar(x, y, params):
    u_xx = jax.grad(lambda x_: u_x_scalar(x_, y, params), argnums=0)(x)
    u_yy = jax.grad(lambda y_: u_y_scalar(x, y_, params), argnums=0)(y)
    return u_xx + u_yy


# ==========================================
# 5. RESIDUOS/PERDAS (formulacao variacional 2D)
# ==========================================
def compute_F_2d(xy_quad_flat, W_flat, V2D_flat, f_fn):
    """F_{kl} = sum_q W_q f(x_q,y_q) v_{kl}(x_q,y_q) (nao depende dos
    parametros da rede, calculado uma unica vez)."""
    f_vals = jax.vmap(lambda pt: f_fn(pt[0], pt[1]))(xy_quad_flat)
    return jnp.einsum("q,q,qk->k", W_flat, f_vals, V2D_flat)


def boundary_loss(params, xy_x0, xy_x1, xy_y0, xy_y1, du_dx_exact, du_dy_exact, g_fn):
    """Penalidade de contorno (Eq. 7): Neumann homogeneo em x=0,x=1,y=1
    (alvo = derivada exata, obtida por autodiff de u_exact_fn) e
    Dirichlet em y=0 (alvo = g_fn(x) = u_exact(x,0)). Soma dos 4 termos,
    mesmo padrao (sem fator 1/4) de pinn_poisson_2d.py / 2d_poisson.py."""
    r_x0 = jax.vmap(lambda pt: u_x_scalar(pt[0], pt[1], params) - du_dx_exact(pt[0], pt[1]))(xy_x0)
    r_x1 = jax.vmap(lambda pt: u_x_scalar(pt[0], pt[1], params) - du_dx_exact(pt[0], pt[1]))(xy_x1)
    r_y0 = jax.vmap(lambda pt: u_scalar(pt[0], pt[1], params) - g_fn(pt[0]))(xy_y0)
    r_y1 = jax.vmap(lambda pt: u_y_scalar(pt[0], pt[1], params) - du_dy_exact(pt[0], pt[1]))(xy_y1)
    return jnp.mean(r_x0 ** 2) + jnp.mean(r_x1 ** 2) + jnp.mean(r_y0 ** 2) + jnp.mean(r_y1 ** 2)


def vpinn_loss_R1(params, xy_quad_flat, W_flat, V2D_flat, F, tau,
                   xy_x0, xy_x1, xy_y0, xy_y1, du_dx_exact, du_dy_exact, g_fn):
    """L^(1): R_{kl}^(1) = sum_q W_q Delta u(x_q,y_q) v_{kl}(x_q,y_q)
    (sem IBP)."""
    lap_vals = jax.vmap(lambda pt: laplacian_scalar(pt[0], pt[1], params))(xy_quad_flat)
    R = jnp.einsum("q,q,qk->k", W_flat, lap_vals, V2D_flat)
    L_R = jnp.mean((R - F) ** 2)
    L_b = boundary_loss(params, xy_x0, xy_x1, xy_y0, xy_y1, du_dx_exact, du_dy_exact, g_fn)
    return L_R + tau * L_b


def vpinn_loss_R2(params, xy_quad_flat, W_flat, Vx2D_flat, Vy2D_flat, F, tau,
                   xy_x0, xy_x1, xy_y0, xy_y1, du_dx_exact, du_dy_exact, g_fn):
    """L^(2): R_{kl}^(2) = -sum_q W_q (u_x v_{kl,x} + u_y v_{kl,y}) (uma
    IBP; o termo de fronteira se anula pois v_{kl} = 0 em toda a
    fronteira de (0,1)^2)."""
    ux_vals = jax.vmap(lambda pt: u_x_scalar(pt[0], pt[1], params))(xy_quad_flat)
    uy_vals = jax.vmap(lambda pt: u_y_scalar(pt[0], pt[1], params))(xy_quad_flat)
    R = -jnp.einsum("q,q,qk->k", W_flat, ux_vals, Vx2D_flat) - jnp.einsum("q,q,qk->k", W_flat, uy_vals, Vy2D_flat)
    L_R = jnp.mean((R - F) ** 2)
    L_b = boundary_loss(params, xy_x0, xy_x1, xy_y0, xy_y1, du_dx_exact, du_dy_exact, g_fn)
    return L_R + tau * L_b


def generate_boundary_points(n_g, key):
    """Pontos de contorno amostrados por LHS, mantidos fixos durante toda
    a otimizacao L-BFGS de uma run (mesmo padrao de 2d_poisson.py /
    svpinn_1d_poisson.py)."""
    sub = int(jax.random.randint(key, (), 0, 2 ** 31 - 1))
    sampler_g = qmc.LatinHypercube(d=1, seed=sub)

    y_x0 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_x0 = jnp.column_stack([jnp.full(n_g, X_LEFT), y_x0])

    y_x1 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_x1 = jnp.column_stack([jnp.full(n_g, X_RIGHT), y_x1])

    x_y0 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_y0 = jnp.column_stack([x_y0, jnp.full(n_g, Y_BOTTOM)])

    x_y1 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_y1 = jnp.column_stack([x_y1, jnp.full(n_g, Y_TOP)])

    return xy_x0, xy_x1, xy_y0, xy_y1


# ==========================================
# 6. OTIMIZADOR L-BFGS (jaxopt.LBFGS)
# ==========================================
def train_lbfgs(params, loss_fn, num_steps, eval_freq, XY_test, U_test,
                 history_size=LBFGS_HISTORY_SIZE, tol=LBFGS_TOL):
    """
    Treina `params` minimizando `loss_fn(params)` via L-BFGS
    (jaxopt.LBFGS), registrando a loss e o erro L2 relativo (contra
    XY_test/U_test, pontos e valores do ground truth achatados) a cada
    `eval_freq` passos.

    Mesmo padrao em dois niveis usado em vpinn_1d_poisson.py /
    svpinn_1d_poisson.py: um jax.lax.fori_loop de eval_freq passos de
    L-BFGS (sem avaliacao) dentro de um jax.lax.scan de
    num_steps // eval_freq blocos (uma avaliacao por bloco). `loss_fn`
    ja fecha sobre os dados fixos da formulacao variacional e da run
    (grid de quadratura, tabelas de teste, F, tau, pontos de contorno),
    por isso o objetivo do L-BFGS depende apenas do vetor de parametros
    achatado.
    """
    p_flat, unflatten_fn = jax.flatten_util.ravel_pytree(params)

    def objective(p_flat_):
        return loss_fn(unflatten_fn(p_flat_))

    lbfgs = jaxopt.LBFGS(fun=objective, maxiter=num_steps, history_size=history_size, tol=tol)
    num_blocks = num_steps // eval_freq

    def block_fn(carry, _):
        p, state = carry

        def inner_step(_, val):
            p_in, state_in = val
            step = lbfgs.update(p_in, state_in)  # OptStep(params, state)
            return step.params, step.state

        p_next, state_next = jax.lax.fori_loop(0, eval_freq, inner_step, (p, state))

        U_nn = forward(XY_test, unflatten_fn(p_next))
        l2_err = jnp.linalg.norm(U_nn - U_test) / jnp.linalg.norm(U_test)
        return (p_next, state_next), (state_next.value, l2_err)

    init_state = lbfgs.init_state(p_flat)
    run_fn = jax.jit(lambda p, s: jax.lax.scan(block_fn, (p, s), xs=None, length=num_blocks))
    (p_final, _), (loss_traj, err_traj) = run_fn(p_flat, init_state)
    return unflatten_fn(p_final), np.asarray(loss_traj), np.asarray(err_traj)


# ==========================================
# 7. GROUND TRUTH (ground_truth_poisson_2d.py) E LOOP PRINCIPAL
# ==========================================
def load_ground_truth(path):
    """Carrega X, Y, U_true gerados por ground_truth_poisson_2d.py e
    usados como referencia para o erro L2 relativo (avaliacao e curva de
    treino)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ground truth '{path}' nao encontrado. Rode "
            f"ground_truth_poisson_2d.py antes deste script."
        )
    with open(path, "r", encoding="utf-8") as f_gt:
        gt_data = json.load(f_gt)
    X = jnp.asarray(gt_data["X"], dtype=jnp.float64)
    Y = jnp.asarray(gt_data["Y"], dtype=jnp.float64)
    U_true = jnp.asarray(gt_data["U_true"], dtype=jnp.float64)
    return X, Y, U_true


X_MESH_GT, Y_MESH_GT, U_TRUE_GT = load_ground_truth(GT_JSON_PATH)
XY_TEST = jnp.stack([X_MESH_GT.ravel(), Y_MESH_GT.ravel()], axis=-1)
U_TEST = U_TRUE_GT.ravel().reshape(-1, 1)

f_fn, du_dx_exact, du_dy_exact = make_force_term_2d(u_exact_fn)
g_fn = lambda x: u_exact_fn(x, jnp.asarray(Y_BOTTOM))

# ------ dados fixos da formulacao variacional (nao dependem da rede) ------
xi_q, x_q, w_q_phys = gauss_legendre_quadrature_unit_interval(Q_QUADRATURE)
V, Vx_phys = make_test_function_tables(xi_q, K_TEST_FUNCTIONS)
V2D_flat, Vx2D_flat, Vy2D_flat = build_2d_test_function_tables(
    V, Vx_phys, Q_QUADRATURE, K_TEST_FUNCTIONS
)
XY_QUAD_FLAT = jnp.stack(jnp.meshgrid(x_q, x_q, indexing="ij"), axis=-1).reshape(-1, 2)
W_FLAT = jnp.outer(w_q_phys, w_q_phys).reshape(-1)
F_2D = compute_F_2d(XY_QUAD_FLAT, W_FLAT, V2D_flat, f_fn)

for L in HIDDEN_LAYER_CONFIGS:
    for NEURONS in NEURONS_PER_LAYER:
        width = [2] + [NEURONS] * L + [1]

        for method_tag in ["R1", "R2"]:
            print("\n" + "=" * 20)
            print(f"PROBLEMA=poisson_2d | METODO={method_tag} | ARQ={width}")
            print("=" * 20)

            loss_trajectories, err_trajectories, l2_errors = [], [], []
            acc_train_time, acc_eval_time = 0.0, 0.0
            U_nn_final, params_final_flat = None, None

            for run in range(NUM_RUNS):
                run_key = random.fold_in(main_key, hash((method_tag, L, run)) % (2 ** 31))
                param_key, boundary_key = random.split(run_key)
                params = init_mlp_params(param_key, width)

                # Pontos de contorno fixos para esta run (nao resampled a
                # cada passo de L-BFGS, mesmo padrao de 2d_poisson.py).
                xy_x0, xy_x1, xy_y0, xy_y1 = generate_boundary_points(N_G_BOUNDARY, boundary_key)

                if method_tag == "R1":
                    loss_fn_run = lambda p: vpinn_loss_R1(
                        p, XY_QUAD_FLAT, W_FLAT, V2D_flat, F_2D, TAU_VPINN,
                        xy_x0, xy_x1, xy_y0, xy_y1, du_dx_exact, du_dy_exact, g_fn
                    )
                else:
                    loss_fn_run = lambda p: vpinn_loss_R2(
                        p, XY_QUAD_FLAT, W_FLAT, Vx2D_flat, Vy2D_flat, F_2D, TAU_VPINN,
                        xy_x0, xy_x1, xy_y0, xy_y1, du_dx_exact, du_dy_exact, g_fn
                    )

                t0 = time.perf_counter()
                params, loss_traj, err_traj = train_lbfgs(
                    params, loss_fn_run, NUM_STEPS, EVAL_FREQ, XY_TEST, U_TEST
                )
                jax.block_until_ready(params)
                t_train = time.perf_counter() - t0
                acc_train_time += t_train

                loss_trajectories.append(loss_traj)
                err_trajectories.append(err_traj)

                _ = forward(XY_TEST, params).block_until_ready()
                t0 = time.perf_counter()
                U_nn = forward(XY_TEST, params)
                U_nn.block_until_ready()
                acc_eval_time += time.perf_counter() - t0

                rel_l2 = float(jnp.linalg.norm(U_nn - U_TEST) / jnp.linalg.norm(U_TEST))
                l2_errors.append(rel_l2)
                print(f"Run {run + 1}/{NUM_RUNS}: erro L2 relativo = {rel_l2:.6e} "
                      f"(treino: {t_train:.2f}s)")

                if run == NUM_RUNS - 1:
                    U_nn_final = np.asarray(U_nn)
                    flat_params, _ = jax.flatten_util.ravel_pytree(params)
                    params_final_flat = np.asarray(flat_params)

            avg_rel_l2 = float(np.mean(l2_errors))
            median_rel_l2 = float(np.median(l2_errors))
            avg_train_time = acc_train_time / NUM_RUNS
            avg_eval_time = acc_eval_time / NUM_RUNS

            print(f"--- Erro L2 relativo medio: {avg_rel_l2:.6e} | "
                  f"mediana: {median_rel_l2:.6e} ---")

            arch_str = "_".join(map(str, width))
            tag = f"poisson_2d_{method_tag}_{arch_str}"

            output_dir = Path("vpinn_poisson_2d")
            output_dir.mkdir(exist_ok=True)

            nome_arquivo = output_dir / f"dados_vpinn_{tag}.json"
            results = {
                "architecture": width,
                "num_hidden_layers": L,
                "method": f"VPINN {method_tag[-2:]} - Legendre (L-BFGS)",
                "problem": "poisson_2d",
                "domain": [[X_LEFT, X_RIGHT], [Y_BOTTOM, Y_TOP]],
                "n_test_functions_per_axis": K_TEST_FUNCTIONS,
                "n_test_functions": K_TEST_FUNCTIONS ** 2,
                "n_quadrature_points_per_axis": Q_QUADRATURE,
                "n_quadrature_points": Q_QUADRATURE ** 2,
                "n_boundary_points": N_G_BOUNDARY,
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
                "x": np.asarray(X_MESH_GT).ravel().tolist(),
                "y": np.asarray(Y_MESH_GT).ravel().tolist(),
                "u_exact": np.asarray(U_TRUE_GT).ravel().tolist(),
                "u_nn": U_nn_final.ravel().tolist(),
                "network_weights": params_final_flat.tolist(),
            }
            with open(nome_arquivo, "w") as fjson:
                json.dump(points, fjson, indent=4)

            nome_arquivo = output_dir / f"curva_treino_vpinn_{tag}.json"
            loss_trajectories = np.stack(loss_trajectories, axis=0)
            err_trajectories = np.stack(err_trajectories, axis=0)
            training_curve = {
                "architecture": width,
                "method": f"VPINN {method_tag[-2:]} - Legendre (L-BFGS)",
                "problem": "poisson_2d",
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

print("\nConcluido. Todos os arquivos JSON foram salvos em vpinn_poisson_2d/.")