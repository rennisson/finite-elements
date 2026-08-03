# -*- coding: utf-8 -*-
"""
Amostra das funções de teste para as SV-PINNs em dominios de hipercubo (0,1)^d.

As funçoes de teste sao realizacoes do processo Whittle-Matern process

    Phi(xi, x) = tau * sum_k (1 + lambda_k)^{-1/2} w_k(xi) phi_k(x)

aproximado sobre o grid Omega = (0,1)^d pela solução da SPDE discretizada

    (1 - Delta_h)^{1/2} Phi^(h) = tau * W

via DST-1 (Transformada de Seno Discreta do Tipo 1)
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

config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

activation_function = jax.nn.tanh


def u(x, y):
    """Solucao exata da EDP"""
    return x ** 2 * (x - 1) ** 2 * y * (y - 1) ** 2


def rhs(x, y):
    """Define o lado direito da equação"""
    return 2 * (x**4 * (3*y - 2) + x**3 * (4 - 6*y) + x**2 * (6*y**3 - 12*y**2 + 9*y - 2)
                - 6*x*(y - 1)**2 * y + (y - 1)**2 * y)


# ==========================================
# 1. CONFIGURAÇÕES DO DOMÍNIO E GROUND TRUTH
# ==========================================
x_lower = 0
x_upper = 1

gt_file = "gt_poisson_2d.json"
if not os.path.exists(gt_file):
    raise FileNotFoundError(f"Arquivo '{gt_file}' não encontrado.")

with open(gt_file, 'r') as f:
    gt_data = json.load(f)

X_mesh_gt = jnp.array(gt_data["X"])
Y_mesh_gt = jnp.array(gt_data["Y"])
U_true_gt = jnp.array(gt_data["U_true"])
xy_points_gt = jnp.stack([X_mesh_gt.ravel(), Y_mesh_gt.ravel()], axis=-1)

print(f"Ground truth carregado. Shape da malha de avaliação: {X_mesh_gt.shape}")


# ==========================================
# 1. DST-I (orthonormal, self-inverse)
# ==========================================
def dst1_matrix(n, dtype=jnp.float64):
    """
    Matriz ortonormal DST-I (n x n), S @ S == I (self-inverse).

    Matriz para 1-dimensão. Consequencia da separabilidade da DST-1.
    Conseguimos usar a mesma matriz e aplicá-la eixo por eixo, ajustando assim
    os custos computacionais desta implementação para niveis razoáveis.

    Parametros
    --------
    n: numero de pontos de colocação

    Retorna
    --------
    S: matrix DST-I (n x n)
    """
    h = 1.0 / (n + 1)
    idx = jnp.arange(1, n + 1, dtype=dtype)
    S = jnp.sqrt(2 * h) * jnp.sin(jnp.pi * jnp.outer(idx, idx) * h)
    return S


def dst1_nd(u_grid, S):
    """
    Aplica a DST ortonormal e d-dimensional em um tensor u, um eixo por vez.
    (autofunções do Laplaciano de Dirichlet em (0,1)^d
    sao produtos de seno em 1 dimensao, pela equacao 4.2)
    """
    out = u_grid
    for axis in range(u_grid.ndim):
        out = jnp.moveaxis(jnp.tensordot(S, out, axes=([1], [axis])), 0, axis)
    return out


# ==========================================
# 2. Discretised eigenvalues of the Dirichlet Laplacian (between Eq. 4.9-4.10)
# ==========================================
def discrete_eigenvalues(n, d, dtype=jnp.float64):
    """
    lambda_k^(h) do Laplaciano Discreto no grid, para o dominio no hipercubo (0,1)^d, k_j = 1..n.

    Parametros
    ----------
    n: numero de pontos de colocação
    d: dimensão

    Retorna
    ----------
    Autovalores discretizados
    """
    h = 1.0 / (n + 1)  # tamanho do passo
    k = jnp.arange(1, n + 1, dtype=dtype)
    lam_1d = (4.0 / (h ** 2)) * (jnp.sin(jnp.pi * k * h / 2.0) ** 2)
    grids = jnp.meshgrid(*([lam_1d] * d), indexing="ij")
    return sum(grids)


# ==========================================
# 3. Sampling realisations of Phi^(h) (Eq. 4.10)
# ==========================================
def sample_test_function(key, n, d, tau=1.0, dtype=jnp.float64):
    """
    Cria uma realização da função de teste Phi^(h) no grid x^(k), k_j = 1,...,n,
    no dominio do hipercubo (0,1)^d (Eq. 4.10).

    Parametros
    ----------
    n: numero de pontos de colocação
    d: dimensão
    tau: fator de escala

    Retorna
    ----------
    Phi(h), função de teste
    """
    h = 1.0 / (n + 1)
    S = dst1_matrix(n, dtype=dtype)
    lam_k = discrete_eigenvalues(n, d, dtype=dtype)
    weight = (1.0 + lam_k) ** (-0.5)

    W = random.normal(key, (n,) * d, dtype=dtype)    # white noise on the grid
    W_hat = dst1_nd(W, S)                             # DST-I of the white noise
    Phi_hat = weight * W_hat                          # solve in spectral space
    Phi = dst1_nd(Phi_hat, S)                          # DST^{-1} = DST-I (self-inverse)
    return (h ** (-d / 2.0)) * tau * Phi


def sample_test_functions(key, n, d, N, tau=1.0, dtype=jnp.float64):
    """
    Constroi N realizações iid phi_1, ..., phi_N de Phi^(h),
    empilhadas ao longo do eixo 0, e shape (N,) + (n,)*d.

    Estas são as funções teste empiricas da perda da SV-PINN. (eq. 4.4).
    """
    # A partir da matrix S, seremos capazes de gerar a DST-1 em n-dimensões,
    # apenas aplicando a S eixo por eixo.
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


# ==========================================
# 4. Grid of collocation points x^(k) in (0,1)^d (matching the test functions)
# ==========================================
def grid_points(n, d, dtype=jnp.float64):
    h = 1.0 / (n + 1)
    coords = (jnp.arange(1, n + 1, dtype=dtype) * h,) * d
    mesh = jnp.meshgrid(*coords, indexing="ij")
    return jnp.stack(mesh, axis=-1)  # shape (n,)*d + (d,)


# ==========================================
# 1. Empirical Phi-stochastically weak norm squared (Eq. 4.4)
# ==========================================
def sv_pinn_norm_squared(residual, phi_samples):
    """
    L_{Phi^(n)}(theta) = (1/N) sum_j [ (1/N_c) sum_i R(x_c^i) phi_j(x_c^i) ]^2

    Parametros
    ----------
    residual: R(x_c^(i)) = L u_theta(x_c^(i)) - f(x_c^(i)) no grid de
              colocacao, shape (n,)*d (mesmo grid dos phi_samples).
    phi_samples: N realizacoes de Phi^(h), shape (N,) + (n,)*d
                 (saida de sample_test_functions).

    Retorna
    ----------
    Escalar, a perda empirica L_{Phi^(n)}(theta) (Eq. 4.4).
    """
    # A tupla nao contempla o eixo 0 (o range começa a partir de 1),
    # porque o eixo 0 indexa qual é a realização N da função de teste phi_j
    # Portanto, a tupla 'axes' guarda apenas os eixos dos pontos de colocação N_c
    axes = tuple(range(1, phi_samples.ndim))  # todos os eixos espaciais

    # média dos produtos internos sobre o eixo dos pontos de colocação
    inner_products = jnp.mean(residual[None, ...] * phi_samples, axis=axes)  # (N,)

    # retorna a media sobre o eixo j (o mesmo que foi ignorado na construção da tupla inicial)
    return jnp.mean(inner_products ** 2)


# ==========================================
# 3. Perda total da SV-PINN (Eq. 4.5)
# ==========================================
def sv_pinn_loss(residual_interior, phi_samples, boundary_residual=None, lam=0.0):
    """
    L_SV(theta) = L_{Phi^(n)}(theta) + lambda * L_b(theta)

    lam = 0 quando a condicao de contorno e' imposta de forma "hard"
    na arquitetura (e.g. via DAFF), como no paper.

    Parametros
    ----------
    residual_interior: R(x_c^(i)) no grid de colocacao, shape (n,)*d.
    phi_samples: realizacoes de Phi^(h), shape (N,) + (n,)*d.
    boundary_residual: u_theta(x_b) - g(x_b) nos pontos de contorno
                        (qualquer shape); None se lam = 0.
    lam: peso da penalidade de contorno.

    Retorna
    ----------
    Escalar, L_SV(theta).
    """
    L_phi = sv_pinn_norm_squared(residual_interior, phi_samples)
    if lam == 0.0 or boundary_residual is None:
        return L_phi
    L_b = jnp.mean(boundary_residual ** 2)
    return L_phi + lam * L_b


# ==========================================
# 5. ARQUITETURA DA REDE E RESÍDUO DA EDP (Poisson 2D, condições mistas)
#    MLP MODIFICADA (Sec. 4.1 do paper / Eq. 4.1), com conexoes residuais
#    multiplicativas: U(x) = sigma(W1 x + b1), V(x) = sigma(W2 x + b2),
#    g^(l) = f^(l) * U + (1 - f^(l)) * V a cada camada oculta.
#    Assume que todas as camadas ocultas tem a MESMA largura r (exigido
#    pela multiplicacao elemento-a-elemento com U, V).
#
#    Aqui devolvemos os RESÍDUOS BRUTOS, pois a perda SV-PINN (Eq. 4.4)
#    precisa do resíduo ponderado pelas funções de teste, não do seu MSE.
#
#    d_n u(0,y) = 0, d_n u(1,y) = 0, u(x,0) = 0, d_n u(x,1) = 0  (Eq. 7)
# ==========================================
def init_mlp_params(key, width, dtype=jnp.float64):
    """width = [d_in, r, r, ..., r, 1] (mesma largura r em todas as camadas
    ocultas). Retorna [encode] + [camada_1, ..., camada_K, saida], onde
    'encode' guarda os pesos de U e V (Eq. 4.1)."""
    initializer = jax.nn.initializers.glorot_normal()
    keys = jax.random.split(key, 4 + (len(width) - 1))
    WU = initializer(keys[0], (width[0], width[1]), dtype)
    BU = initializer(keys[1], (1, width[1]), dtype)
    WV = initializer(keys[2], (width[0], width[1]), dtype)
    BV = initializer(keys[3], (1, width[1]), dtype)
    params = [{'WU': WU, 'BU': BU, 'WV': WV, 'BV': BV}]
    for k, lin, lout in zip(keys[4:], width[:-1], width[1:]):
        W = initializer(k, (lin, lout), dtype)
        B = initializer(k, (1, lout), dtype)
        params.append({'W': W, 'B': B})
    return params


@jax.jit
def forward(xy, params):
    encode, *hidden, output = params
    U = activation_function(xy @ encode['WU'] + encode['BU'])
    V = activation_function(xy @ encode['WV'] + encode['BV'])
    x = xy
    for layer in hidden:
        f = activation_function(x @ layer['W'] + layer['B'])
        x = f * U + (1 - f) * V
    return x @ output['W'] + output['B']


def u_net(x, y, params):
    xy = jnp.array([[x, y]])
    return forward(xy, params)[0, 0]


def laplacian_u(x, y, params):
    u_xx = jax.grad(jax.grad(lambda x_: u_net(x_, y, params), 0), 0)(x)
    u_yy = jax.grad(jax.grad(lambda y_: u_net(x, y_, params), 0), 0)(y)
    return u_xx + u_yy


def du_dx(x, y, params):
    return jax.grad(lambda x_: u_net(x_, y, params), 0)(x)


def du_dy(x, y, params):
    return jax.grad(lambda y_: u_net(x, y_, params), 0)(y)


def residual_interior(params, xy_grid):
    """R(x_c^(i)) = Delta u_theta(x_c^(i)) - rhs(x_c^(i)) no grid de colocação
    xy_grid, shape (n, n, 2) -> retorna shape (n, n), o mesmo grid usado para
    amostrar phi_samples (d = 2)."""
    def residual_point(xy):
        x, y = xy[0], xy[1]
        return laplacian_u(x, y, params) - rhs(x, y)

    flat = xy_grid.reshape(-1, 2)
    R_flat = jax.vmap(residual_point)(flat)
    return R_flat.reshape(xy_grid.shape[:-1])


def boundary_residual(params, xy_x0, xy_x1, xy_y0, xy_y1):
    """Concatena os 4 resíduos de contorno da Eq. (7) (todos com alvo 0):
    d_n u(0,y), d_n u(1,y), u(x,0), d_n u(x,1)."""
    r_x0 = jax.vmap(lambda pt: du_dx(pt[0], pt[1], params))(xy_x0)
    r_x1 = jax.vmap(lambda pt: du_dx(pt[0], pt[1], params))(xy_x1)
    r_y0 = jax.vmap(lambda pt: u_net(pt[0], pt[1], params))(xy_y0)
    r_y1 = jax.vmap(lambda pt: du_dy(pt[0], pt[1], params))(xy_y1)
    return jnp.concatenate([r_x0, r_x1, r_y0, r_y1])


def generate_boundary_points(n_g, key):
    """Pontos de contorno amostrados por LHS, mantidos fixos durante toda a
    otimização (analogamente às funções teste, Sec. 4.5), diferente do
    resampling a cada passo usado no treino por Adam em pinn_poisson_2d.py."""
    sub = int(jax.random.randint(key, (), 0, 2**31 - 1))
    sampler_g = qmc.LatinHypercube(d=1, seed=sub)

    y_x0 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_x0 = jnp.column_stack([jnp.zeros(n_g), y_x0])

    y_x1 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_x1 = jnp.column_stack([jnp.ones(n_g), y_x1])

    x_y0 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_y0 = jnp.column_stack([x_y0, jnp.zeros(n_g)])

    x_y1 = jnp.array(sampler_g.random(n=n_g), dtype=jnp.float64).flatten()
    xy_y1 = jnp.column_stack([x_y1, jnp.ones(n_g)])

    return xy_x0, xy_x1, xy_y0, xy_y1


# ==========================================
# 6a. CALIBRAÇÃO DE TAU (Eq. 6.1)
#    tau^2 = L_b(theta_0) / L_{1,Phi^(n)}(theta_0), com lambda = 1
#    (equilibra os dois termos da perda na inicialização; Phi_tau = tau * Phi_{tau=1}
#    e' linear em tau (Eq. 3.2/4.6), entao basta reescalar phi_samples_tau1)
# ==========================================
def calibrate_tau(params0, xy_grid, phi_samples_tau1, xy_x0, xy_x1, xy_y0, xy_y1):
    R0 = residual_interior(params0, xy_grid)
    L1_phi0 = sv_pinn_norm_squared(R0, phi_samples_tau1)
    Rb0 = boundary_residual(params0, xy_x0, xy_x1, xy_y0, xy_y1)
    Lb0 = jnp.mean(Rb0 ** 2)
    tau2 = Lb0 / L1_phi0
    return jnp.sqrt(tau2)


# ==========================================
# 6b. CALIBRAÇÃO (E ATUALIZAÇÃO) DE LAMBDA — MANTIDO COMO ESTAVA
#    lambda = L_Phi(theta) / L_b(theta)
#    (equilibra os dois termos da perda (Eq. 4.5) nos parametros atuais;
#    recalculado periodicamente ao longo do treinamento, acompanhando theta)
# ==========================================
def compute_lambda(params, xy_grid, phi_samples, xy_x0, xy_x1, xy_y0, xy_y1, eps=1e-10):
    """eps evita divisao por L_b quase-zero (observado a levar lambda a
    explodir e o treinamento a divergir quando o contorno ja' esta' bem
    ajustado mas o interior ainda nao)."""
    R = residual_interior(params, xy_grid)
    L_phi = sv_pinn_norm_squared(R, phi_samples)
    Rb = boundary_residual(params, xy_x0, xy_x1, xy_y0, xy_y1)
    L_b = jnp.mean(Rb ** 2)
    return L_phi / (L_b + eps)


# ==========================================
# 7. TREINAMENTO DAS SV-PINNs (SOMENTE L-BFGS, SEM ADAM)
# ==========================================
architectures = [
    [20, 1], [60, 1],
    [20, 20, 1], [60, 60, 1],
    [20, 20, 20, 1], [60, 60, 60, 1],
    [20, 20, 20, 20, 1], [60, 60, 60, 60, 1],
    [20, 20, 20, 20, 20, 1], [60, 60, 60, 60, 60, 1],
    [120, 120, 120, 120, 120, 1]
]

n_colloc = 128              # pontos de colocação por eixo = grid da DST-I (Table A.8, caso 2D)
N_test_functions = 25000    # numero de funcoes teste amostradas (Table A.8, caso 2D)
n_g_boundary = 250          # pontos de contorno por aresta (N_g de pinn_poisson_2d.py)
num_runs = 3                 # 3 repeticoes, como no protocolo experimental do paper (Sec. 6)
lbfgs_maxiter = 5000          # SV-PINNs treinadas por L-BFGS por 5,000 passos (Sec. 6)
n_lambda_updates = 5           # numero de vezes que lambda e recalculado durante o treinamento

main_key = jax.random.PRNGKey(42)

# grid de colocacao x_c^(i,j), fixo, identico ao grid usado para amostrar phi (Sec. 4.3.1)
xy_grid = grid_points(n_colloc, d=2)

for arch in architectures:
    width = [2] + arch
    arch_str = "_".join(map(str, width))

    print("\n" + "=" * 50)
    print(f"INICIANDO: ARQUITETURA {width} | {num_runs} EXECUÇÕES (SV-PINN, L-BFGS)")
    print("=" * 50)

    acc_train_lbfgs = 0.0
    acc_eval_time = 0.0
    l2_errors = []
    l2_error_runs = []   # trajetoria (por passo) de erro L2, uma lista por run
    loss_runs = []        # trajetoria (por passo) da perda, uma lista por run

    U_nn_final = None
    params_final_flat = None

    initializer = jax.nn.initializers.glorot_normal()
    dummy_params = init_mlp_params(main_key, width, dtype=jnp.float64)
    dummy_params = jax.tree_util.tree_map(jnp.zeros_like, dummy_params)

    _, unflatten_fn = jax.flatten_util.ravel_pytree(dummy_params)

    @jax.jit
    def objective_lbfgs(p_flat, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1, lam):
        # A avaliacao da métrica L2 (que consome muito tempo) foi removida do objetivo.
        params = unflatten_fn(p_flat)
        R = residual_interior(params, xy_grid)
        Rb = boundary_residual(params, xy_x0, xy_x1, xy_y0, xy_y1)
        L_phi = sv_pinn_norm_squared(R, phi_samples_run)
        L_b = jnp.mean(Rb ** 2)
        return L_phi + lam * L_b

    segment_iters = max(1, lbfgs_maxiter // n_lambda_updates)
    
    # maxiter definido para segment_iters; has_aux removido
    lbfgs = jaxopt.LBFGS(fun=objective_lbfgs, maxiter=segment_iters,
                          history_size=50, tol=1e-12)

    # ---------------------------------------------------------
    # NOVIDADE: Segmento JIT-compilado para rodar o L-BFGS em blocos
    # ---------------------------------------------------------
    @jax.jit
    def lbfgs_segment_with_trajectory(p_flat, state, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1, lam):
        eval_freq = 10
        num_blocks = segment_iters // eval_freq

        def block_fn(carry, _):
            p, s = carry

            # Loop interno: executa 10 iteracoes apenas atualizando os pesos (sem inferencia)
            def inner_step(i, val):
                p_in, s_in = val
                p_out, s_out = lbfgs.update(p_in, s_in,
                                            phi_samples_run=phi_samples_run,
                                            xy_x0=xy_x0, xy_x1=xy_x1,
                                            xy_y0=xy_y0, xy_y1=xy_y1,
                                            lam=lam)
                return p_out, s_out

            p_next, s_next = jax.lax.fori_loop(0, eval_freq, inner_step, (p, s))

            # Avaliacao: ocorre apenas 1x ao final do bloco de 10 passos
            params_ = unflatten_fn(p_next)
            u_nn = forward(xy_points_gt, params_).reshape(X_mesh_gt.shape)
            l2_err = jnp.linalg.norm(u_nn - U_true_gt) / jnp.linalg.norm(U_true_gt)

            # state.value guarda o ultimo valor da Loss calculado
            return (p_next, s_next), (s_next.value, l2_err)

        (p_final, state_final), (loss_traj, err_traj) = jax.lax.scan(
            block_fn, (p_flat, state), xs=None, length=num_blocks)

        return p_final, state_final, loss_traj, err_traj


    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")

        # 1. Inicializar parâmetros
        run_key = jax.random.fold_in(main_key, run)
        params = init_mlp_params(run_key, width, dtype=jnp.float64)
        params_flat, _ = jax.flatten_util.ravel_pytree(params)

        # 2. Amostrar as N funções teste e os pontos de contorno
        run_key, sub_phi, sub_bc = jax.random.split(run_key, 3)
        phi_samples_tau1 = sample_test_functions(sub_phi, n=n_colloc, d=2, N=N_test_functions, tau=1.0)
        xy_x0, xy_x1, xy_y0, xy_y1 = generate_boundary_points(n_g_boundary, sub_bc)

        # 3. Calibrar tau
        tau_run = calibrate_tau(params, xy_grid, phi_samples_tau1, xy_x0, xy_x1, xy_y0, xy_y1)
        phi_samples_run = tau_run * phi_samples_tau1

        # 4. Lambda inicial
        lam = compute_lambda(params, xy_grid, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1)

        # 5. Otimização via L-BFGS preservando a continuidade do estado 'state'
        state = lbfgs.init_state(params_flat, phi_samples_run=phi_samples_run,
                                  xy_x0=xy_x0, xy_x1=xy_x1, xy_y0=xy_y0, xy_y1=xy_y1, lam=lam)

        run_l2_traj_segments = []
        run_loss_traj_segments = []

        t_start_lbfgs = time.perf_counter()
        
        for seg in range(n_lambda_updates):
            # Passa o 'state' preservado do segmento anterior
            params_flat, state, loss_traj_seg, err_traj_seg = lbfgs_segment_with_trajectory(
                params_flat, state, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1, lam)

            run_l2_traj_segments.append(np.asarray(err_traj_seg))
            run_loss_traj_segments.append(np.asarray(loss_traj_seg))

            # Atualiza lambda para o proximo segmento
            if seg < n_lambda_updates - 1:
                params = unflatten_fn(params_flat)
                lam = compute_lambda(params, xy_grid, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1)

        run_l2_traj = np.concatenate(run_l2_traj_segments)
        run_loss_traj = np.concatenate(run_loss_traj_segments)

        params = unflatten_fn(params_flat).copy()
        loss_lbfgs = run_loss_traj[-1]
        t_lbfgs = time.perf_counter() - t_start_lbfgs

        l2_error_runs.append(run_l2_traj)
        loss_runs.append(run_loss_traj)

        acc_train_lbfgs += t_lbfgs
        print(f"L-BFGS concluído em {t_lbfgs:.2f}s | tau = {float(tau_run):.4e} | "
              f"lambda final = {float(lam):.4e} | Loss: {loss_lbfgs:.8e}")

        # 6. Avaliação nos pontos Ground Truth
        t_start_eval = time.perf_counter()
        u_nn_flat = forward(xy_points_gt, params)
        u_nn = u_nn_flat.reshape(X_mesh_gt.shape)
        u_nn.block_until_ready()
        t_eval = time.perf_counter() - t_start_eval
        acc_eval_time += t_eval

        rel_l2_error = run_l2_traj[-1]
        l2_errors.append(rel_l2_error)
        print(f"Erro L2 Relativo (Run {run + 1}): {rel_l2_error:.8e}")

        if run == num_runs - 1:
            U_nn_final = np.asarray(u_nn)
            params_final_flat = np.asarray(params_flat)

    avg_train_lbfgs = float(acc_train_lbfgs / num_runs)
    avg_eval_time = float(acc_eval_time / num_runs)
    avg_rel_l2 = float(np.mean(l2_errors))
    median_rel_l2 = float(np.median(l2_errors))

    print("\n" + "-" * 30)
    print(f"RESUMO DAS MÉDIAS ({num_runs} RUNS) - ARQUITETURA {width}")
    print(f"Erro L2 Relativo Médio: {avg_rel_l2:.8e}")
    print(f"Tempo Médio Treino L-BFGS: {avg_train_lbfgs:.3f}s")
    print(f"Tempo Médio Avaliação: {avg_eval_time:.5f}s")
    print("-" * 30)

    results = {
        'architecture': width,
        'num_hidden_layers': len(width) - 1,
        'method': 'SV-PINN 2D (L-BFGS)',
        'n_colloc': n_colloc,
        'n_test_functions': N_test_functions,
        'lbfgs_maxiter': lbfgs_maxiter,
        'num_runs_avg': num_runs,
        'error_relativo_medio': avg_rel_l2,
        'error_relativo_mediana': median_rel_l2,
        'time_training_lbfgs': avg_train_lbfgs,
        'time_evaluation': avg_eval_time,
        'num_params': int(len(params_final_flat))
    }

    nome_arquivo = f'dados_svpinn_2d_{arch_str}.json'
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    points = {
        'x_mesh': np.asarray(X_mesh_gt).tolist(),
        'y_mesh': np.asarray(Y_mesh_gt).tolist(),
        'u_nn': U_nn_final.tolist(),
        'network_weights': params_final_flat.tolist()
    }

    nome_arquivo_pontos = f'pontos_svpinn_2d_{arch_str}.json'
    with open(nome_arquivo_pontos, 'w', encoding='utf-8') as f:
        json.dump(points, f, indent=4)

    # ---------------------------------------------------------
    # NOVIDADE: Exportação da curva seguindo o mesmo layout (1D)
    # ---------------------------------------------------------
    all_loss_trajectories = np.stack(loss_runs, axis=0)         
    all_l2_error_trajectories = np.stack(l2_error_runs, axis=0) 

    training_curve = {
        'architecture': width,
        'method': 'SV-PINN 2D (L-BFGS)',
        'n_colloc': n_colloc,
        'n_test_functions': N_test_functions,
        'lbfgs_maxiter': lbfgs_maxiter,
        'num_runs': num_runs,
        # Registrado a cada 10 passos
        'steps': list(range(10, lbfgs_maxiter + 1, 10)),
        'l2_relative_error_per_run': all_l2_error_trajectories.tolist(),
        'loss_per_run': all_loss_trajectories.tolist(),
        'l2_relative_error_mean': all_l2_error_trajectories.mean(axis=0).tolist(),
        'l2_relative_error_std': all_l2_error_trajectories.std(axis=0).tolist(),
    }

    # Salvo como 'curva_treino_svpinn_2d_*.json' para ser iterado no script gerador do plot
    nome_arquivo_curva = f'curva_treino_svpinn_2d_{arch_str}.json'
    with open(nome_arquivo_curva, 'w', encoding='utf-8') as f:
        json.dump(training_curve, f, indent=4)

    print(f"Dados salvos como: {nome_arquivo}, {nome_arquivo_pontos} e {nome_arquivo_curva}\n")