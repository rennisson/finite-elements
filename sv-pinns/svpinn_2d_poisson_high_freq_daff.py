# -*- coding: utf-8 -*-
"""
Amostra das funções de teste para as SV-PINNs em dominios de hipercubo (0,1)^d.

As funçoes de teste sao realizacoes do processo Whittle-Matern process

    Phi(xi, x) = tau * sum_k (1 + lambda_k)^{-1/2} w_k(xi) phi_k(x)

aproximado sobre o grid Omega = (0,1)^d pela solução da SPDE discretizada

    (1 - Delta_h)^{1/2} Phi^(h) = tau * W

via DST-1 (Transformada de Seno Discreta do Tipo 1)

Experimento 2 (Sec. 6.2): Equação de Poisson 2D multi-escala de alta frequência (Eq. 6.4)
    -Delta u(x, y) = f(x, y),  (x, y) in Omega = (0, 1)^2
    u(x, y) = 0,               (x, y) in dOmega
com solução exata:
    u(x, y) = sin(pi * x) * sin(pi * y) * [sin(a * (x + y)) + sin(2 * pi * x) + cos(3 * pi * y)]
"""
import jax
import jax.flatten_util
import jax.numpy as jnp
import jaxopt
import json
import numpy as np
import os
import sys
import time

from jax import random, config
from scipy.stats import qmc
from pathlib import Path

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

activation_function = jax.nn.tanh

# Parâmetro de frequência da EDP (Eq. 6.4): a in {1, 10, 25, 50, 75}
a = 75

architectures = [
    # [264, 264, 1],
    # [264, 264, 264, 1],
    [512, 512, 1],
    [512, 512, 512, 1]
]

n_colloc = 128              # pontos de colocação por eixo = grid da DST-I (Table A.8, caso 2D)
N_test_functions = 15000    # numero de funcoes teste amostradas (Table A.8, caso 2D)
n_g_boundary = 120          # pontos de contorno por aresta (N_g de pinn_poisson_2d.py)
num_runs = 3
lbfgs_maxiter = 5000       # SV-PINNs treinadas por L-BFGS


def u(x, y, a=a):
    """Solucao exata da EDP (Eq. 6.4)"""
    return (
        jnp.sin(jnp.pi * x)
        * jnp.sin(jnp.pi * y)
        * (jnp.sin(a * (x + y)) + jnp.sin(2.0 * jnp.pi * x) + jnp.cos(3.0 * jnp.pi * y))
    )


def rhs(x, y, a=a):
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
# 1. CONFIGURAÇÕES DO DOMÍNIO E GROUND TRUTH
# ==========================================
x_lower = 0
x_upper = 1

# Carrega o Ground Truth gerado por sv-pinns/ground_truth_poisson_2d_high_freq.py
gt_filename = f"gt_poisson_2d_high_freq_a{a}.json"
gt_candidates = [
    Path(gt_filename),
    Path(__file__).resolve().parent / gt_filename,
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
xy_points_gt = jnp.stack([X_mesh_gt.ravel(), Y_mesh_gt.ravel()], axis=-1)

print(f"Ground truth '{gt_path}' carregado (a = {a}). Shape da malha de avaliação: {X_mesh_gt.shape}")


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
    Phi = dst1_nd(Phi_hat, S)                         # DST^{-1} = DST-I (self-inverse)
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
# 5. DAFF (DOMAIN-AWARE FOURIER FEATURES), ARQUITETURA DA REDE E RESÍDUO DA EDP
#    Codificação DAFF (Sec. 4.1.1, Eq. 4.2, d=2):
#        phi_{k1, k2}(x, y) = 2 * sin(k1 * pi * x) * sin(k2 * pi * y)
#    Selecionam-se as n_daff = 64 * d = 128 autofunções do Laplaciano de
#    Dirichlet em (0, 1)^2 com os menores autovalores lambda_{k1, k2} = pi^2 * (k1^2 + k2^2)
#    dentre k1, k2 <= 64 (Sec. 6 / Table A.8).
#
#    MLP MODIFICADA (Sec. 4.1 do paper / Eq. 4.1), com conexoes residuais
#    multiplicativas alimentadas pelas características DAFF:
#    U(x) = sigma(W1 phi_I(x) + b1), V(x) = sigma(W2 phi_I(x) + b2),
#    g^(l) = f^(l) * U + (1 - f^(l)) * V a cada camada oculta.
#
#    Aqui devolvemos os RESÍDUOS BRUTOS, pois a perda SV-PINN (Eq. 4.4)
#    precisa do resíduo ponderado pelas funções de teste, não do seu MSE.
#
#    u(0,y) = 0, u(1,y) = 0, u(x,0) = 0, u(x,1) = 0  (Eq. 6.4)
# ==========================================
n_daff = 128  # número de funções de base DAFF em 2D (Table A.8: 64 * d = 128)


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
    xy = daff_encoding(xy)
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
        return laplacian_u(x, y, params) - rhs(x, y, a=a)

    flat = xy_grid.reshape(-1, 2)
    R_flat = jax.vmap(residual_point)(flat)
    return R_flat.reshape(xy_grid.shape[:-1])


def boundary_residual(params, xy_x0, xy_x1, xy_y0, xy_y1):
    """Concatena os 4 resíduos de contorno da Eq. (6.4) (todos Dirichlet com alvo 0):
    u(0,y), u(1,y), u(x,0), u(x,1)."""
    r_x0 = jax.vmap(lambda pt: u_net(pt[0], pt[1], params))(xy_x0)
    r_x1 = jax.vmap(lambda pt: u_net(pt[0], pt[1], params))(xy_x1)
    r_y0 = jax.vmap(lambda pt: u_net(pt[0], pt[1], params))(xy_y0)
    r_y1 = jax.vmap(lambda pt: u_net(pt[0], pt[1], params))(xy_y1)
    return jnp.concatenate([r_x0, r_x1, r_y0, r_y1])


def compute_boundary_losses(params, xy_x0, xy_x1, xy_y0, xy_y1):
    """Retorna os resíduos quadráticos médios (MSE) de cada condição de fronteira da Eq. (6.4):
    x=0 (Dirichlet), x=1 (Dirichlet), y=0 (Dirichlet), y=1 (Dirichlet)."""
    r_x0 = jax.vmap(lambda pt: u_net(pt[0], pt[1], params))(xy_x0)
    r_x1 = jax.vmap(lambda pt: u_net(pt[0], pt[1], params))(xy_x1)
    r_y0 = jax.vmap(lambda pt: u_net(pt[0], pt[1], params))(xy_y0)
    r_y1 = jax.vmap(lambda pt: u_net(pt[0], pt[1], params))(xy_y1)
    return (
        jnp.mean(r_x0 ** 2),
        jnp.mean(r_x1 ** 2),
        jnp.mean(r_y0 ** 2),
        jnp.mean(r_y1 ** 2),
    )


def print_boundary_losses(params, xy_x0, xy_x1, xy_y0, xy_y1):
    """Perda MSE de cada condicao de fronteira separadamente (Eq. 6.4),
    para comparacao rapida."""
    loss_x0, loss_x1, loss_y0, loss_y1 = compute_boundary_losses(params, xy_x0, xy_x1, xy_y0, xy_y1)
    labels = ["x=0 (Dirichlet)", "x=1 (Dirichlet)", "y=0 (Dirichlet)", "y=1 (Dirichlet)"]
    losses = [loss_x0, loss_x1, loss_y0, loss_y1]
    print("  Perda residual por condicao de fronteira:")
    for label, val in zip(labels, losses):
        print(f"    {label:<18s}: {float(val):.6e}")


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
    return 2 * (L_phi / (L_b + eps))


# ==========================================
# 7. TREINAMENTO DAS SV-PINNs (SOMENTE L-BFGS, SEM ADAM)
# ==========================================

main_key = jax.random.PRNGKey(42)

# grid de colocacao x_c^(i,j), fixo, identico ao grid usado para amostrar phi (Sec. 4.3.1)
xy_grid = grid_points(n_colloc, d=2)

for arch in architectures:
    width = [2] + arch
    layer_dims = [n_daff] + arch  # primeira camada recebe as `n_daff` características DAFF
    arch_str = "_".join(map(str, width))

    print("\n" + "=" * 50)
    print(f"INICIANDO: ARQUITETURA {width} (DAFF={n_daff}) | a = {a} | {num_runs} EXECUÇÕES (SV-PINN, L-BFGS)")
    print("=" * 50)

    acc_train_lbfgs = 0.0
    acc_eval_time = 0.0
    l2_errors = []
    l2_error_runs = []     # trajetoria (por passo) de erro L2, uma lista por run
    loss_runs = []          # trajetoria (por passo) da perda, uma lista por run
    interior_mse_runs = []  # trajetoria MSE residual do interior
    bc_x0_runs = []         # trajetoria MSE contorno x=0
    bc_x1_runs = []         # trajetoria MSE contorno x=1
    bc_y0_runs = []         # trajetoria MSE contorno y=0
    bc_y1_runs = []         # trajetoria MSE contorno y=1

    U_nn_final = None
    params_final_flat = None

    initializer = jax.nn.initializers.glorot_normal()
    dummy_params = init_mlp_params(main_key, layer_dims, dtype=jnp.float64)
    dummy_params = jax.tree_util.tree_map(jnp.zeros_like, dummy_params)

    _, unflatten_fn = jax.flatten_util.ravel_pytree(dummy_params)

    @jax.jit
    def raw_objective_lbfgs(p_flat, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1, lam):
        # A avaliacao da métrica L2 (que consome muito tempo) foi removida do objetivo.
        params = unflatten_fn(p_flat)
        R = residual_interior(params, xy_grid)
        Rb = boundary_residual(params, xy_x0, xy_x1, xy_y0, xy_y1)
        L_phi = sv_pinn_norm_squared(R, phi_samples_run)
        L_b = jnp.mean(Rb ** 2)
        return L_phi + lam * L_b

    @jax.jit
    def objective_lbfgs(p_flat, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1, lam, loss0):
        # Normaliza pelo valor inicial L(theta_0) para manter os gradientes e a line search bem condicionados
        return raw_objective_lbfgs(p_flat, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1, lam) / loss0

    # lambda e' fixado no inicio de cada run (lambda = L_Phi(theta_0) / L_b(theta_0))
    # e mantido constante durante todo o treinamento; o L-BFGS roda uma unica vez
    # por lbfgs_maxiter passos (sem recalibracao de lambda).
    lbfgs = jaxopt.LBFGS(fun=objective_lbfgs, maxiter=lbfgs_maxiter,
                         history_size=200, tol=1e-12)

    # ---------------------------------------------------------
    # Roda o L-BFGS em blocos apenas para registrar a trajetoria (erro L2,
    # MSE do interior e perdas de contorno a cada 100 passos); lambda permanece fixo.
    # ---------------------------------------------------------
    @jax.jit
    def lbfgs_segment_with_trajectory(p_flat, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1, lam, loss0):
        state = lbfgs.init_state(p_flat, phi_samples_run=phi_samples_run,
                                 xy_x0=xy_x0, xy_x1=xy_x1, xy_y0=xy_y0, xy_y1=xy_y1,
                                 lam=lam, loss0=loss0)

        eval_freq = 100
        num_blocks = lbfgs_maxiter // eval_freq

        def block_fn(carry, _):
            p, s = carry

            # Loop interno: executa eval_freq iteracoes apenas atualizando os pesos (sem inferencia pesada)
            def inner_step(i, val):
                p_in, s_in = val
                p_out, s_out = lbfgs.update(p_in, s_in,
                                            phi_samples_run=phi_samples_run,
                                            xy_x0=xy_x0, xy_x1=xy_x1,
                                            xy_y0=xy_y0, xy_y1=xy_y1,
                                            lam=lam, loss0=loss0)
                return p_out, s_out

            p_next, s_next = jax.lax.fori_loop(0, eval_freq, inner_step, (p, s))

            # Avaliacao: ocorre 1x ao final de cada bloco de passos
            params_ = unflatten_fn(p_next)
            u_nn = forward(xy_points_gt, params_).reshape(X_mesh_gt.shape)
            l2_err = jnp.linalg.norm(u_nn - U_true_gt) / jnp.linalg.norm(U_true_gt)

            # MSE residual do interior (EDP) e de cada condicao de contorno
            r_int = residual_interior(params_, xy_grid)
            interior_mse = jnp.mean(r_int ** 2)
            bc_x0, bc_x1, bc_y0, bc_y1 = compute_boundary_losses(params_, xy_x0, xy_x1, xy_y0, xy_y1)

            # state.value guarda o ultimo valor da Loss (normalizada) calculado
            return (p_next, s_next), (s_next.value, l2_err, interior_mse, bc_x0, bc_x1, bc_y0, bc_y1)

        (p_final, state_final), (
            loss_traj, err_traj, interior_mse_traj, bc_x0_traj, bc_x1_traj, bc_y0_traj, bc_y1_traj
        ) = jax.lax.scan(block_fn, (p_flat, state), xs=None, length=num_blocks)

        return (p_final, state_final, loss_traj, err_traj,
                interior_mse_traj, bc_x0_traj, bc_x1_traj, bc_y0_traj, bc_y1_traj)

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")

        # 1. Inicializar parâmetros
        run_key = jax.random.fold_in(main_key, run)
        params = init_mlp_params(run_key, layer_dims, dtype=jnp.float64)
        params_flat, _ = jax.flatten_util.ravel_pytree(params)

        # 2. Amostrar as N funções teste e os pontos de contorno
        run_key, sub_phi, sub_bc = jax.random.split(run_key, 3)
        phi_samples_tau1 = sample_test_functions(sub_phi, n=n_colloc, d=2, N=N_test_functions, tau=1.0)
        xy_x0, xy_x1, xy_y0, xy_y1 = generate_boundary_points(n_g_boundary, sub_bc)

        # 3. Calibrar tau
        tau_run = calibrate_tau(params, xy_grid, phi_samples_tau1, xy_x0, xy_x1, xy_y0, xy_y1)
        phi_samples_run = tau_run * phi_samples_tau1

        # 4. Lambda fixo: lambda = L_Phi(theta_0) / L_b(theta_0), calculado uma unica
        #    vez a partir dos parametros iniciais e mantido constante durante o treino.
        lam = compute_lambda(params, xy_grid, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1)

        # Valor inicial da perda para normalizar a função objetivo (L(theta_0) -> 1.0)
        loss0 = raw_objective_lbfgs(params_flat, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1, lam)

        # 5. Otimização via L-BFGS, de uma unica vez, por lbfgs_maxiter passos (lambda fixo)
        t_start_lbfgs = time.perf_counter()

        (params_flat, state, run_loss_traj, run_l2_traj, run_interior_mse_traj,
         run_bc_x0_traj, run_bc_x1_traj, run_bc_y0_traj, run_bc_y1_traj) = lbfgs_segment_with_trajectory(
            params_flat, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1, lam, loss0)
        run_loss_traj = np.asarray(run_loss_traj)
        run_l2_traj = np.asarray(run_l2_traj)
        run_interior_mse_traj = np.asarray(run_interior_mse_traj)
        run_bc_x0_traj = np.asarray(run_bc_x0_traj)
        run_bc_x1_traj = np.asarray(run_bc_x1_traj)
        run_bc_y0_traj = np.asarray(run_bc_y0_traj)
        run_bc_y1_traj = np.asarray(run_bc_y1_traj)

        params = unflatten_fn(params_flat)
        loss_lbfgs = objective_lbfgs(
            params_flat, phi_samples_run, xy_x0, xy_x1, xy_y0, xy_y1, lam, loss0
        ).block_until_ready()
        t_lbfgs = time.perf_counter() - t_start_lbfgs

        l2_error_runs.append(run_l2_traj)
        loss_runs.append(run_loss_traj)
        interior_mse_runs.append(run_interior_mse_traj)
        bc_x0_runs.append(run_bc_x0_traj)
        bc_x1_runs.append(run_bc_x1_traj)
        bc_y0_runs.append(run_bc_y0_traj)
        bc_y1_runs.append(run_bc_y1_traj)

        acc_train_lbfgs += t_lbfgs
        print(f"L-BFGS concluído em {t_lbfgs:.2f}s | tau = {float(tau_run):.4e} | "
              f"lambda (fixo) = {float(lam):.4e} | Loss0 = {float(loss0):.4e} | Loss (norm): {loss_lbfgs:.8e}")
        print(f"  MSE residual do interior: {float(run_interior_mse_traj[-1]):.6e}")
        print_boundary_losses(params, xy_x0, xy_x1, xy_y0, xy_y1)

        # 6. Avaliação nos pontos Ground Truth
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

    output_dir = Path(f"svpinn_daff_poisson_2d_high_freq_a{a}")
    output_dir.mkdir(exist_ok=True)

    results = {
        'architecture': width,
        'num_hidden_layers': len(width) - 1,
        'method': 'SV-PINN 2D (L-BFGS)',
        'features': 'DAFF',
        'n_daff': n_daff,
        'a': a,
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

    nome_arquivo = output_dir / f'dados_svpinn_2d_a{a}_{arch_str}.json'
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    points = {
        'x': X_mesh_gt.ravel().tolist(),
        'y': Y_mesh_gt.ravel().tolist(),
        'u_nn': U_nn_final.ravel().tolist(),
        'network_weights': params_final_flat.tolist()
    }

    nome_arquivo_pontos = output_dir / f'pontos_svpinn_2d_a{a}_{arch_str}.json'
    with open(nome_arquivo_pontos, 'w', encoding='utf-8') as f:
        json.dump(points, f, indent=4)

    # ---------------------------------------------------------
    # Exportação das trajetórias (L2, Loss, Interior e Condições de Contorno)
    # ---------------------------------------------------------
    all_loss_trajectories = np.stack(loss_runs, axis=0)
    all_l2_error_trajectories = np.stack(l2_error_runs, axis=0)
    all_interior_mse_trajectories = np.stack(interior_mse_runs, axis=0)
    all_bc_x0_trajectories = np.stack(bc_x0_runs, axis=0)
    all_bc_x1_trajectories = np.stack(bc_x1_runs, axis=0)
    all_bc_y0_trajectories = np.stack(bc_y0_runs, axis=0)
    all_bc_y1_trajectories = np.stack(bc_y1_runs, axis=0)

    training_curve = {
        'architecture': width,
        'method': 'SV-PINN 2D (L-BFGS)',
        'features': 'DAFF',
        'n_daff': n_daff,
        'a': a,
        'n_colloc': n_colloc,
        'n_test_functions': N_test_functions,
        'lbfgs_maxiter': lbfgs_maxiter,
        'num_runs': num_runs,
        # Registrado a cada 100 passos
        'steps': list(range(100, lbfgs_maxiter + 1, 100)),
        'l2_relative_error_per_run': all_l2_error_trajectories.tolist(),
        'loss_per_run': all_loss_trajectories.tolist(),
        'l2_relative_error_mean': all_l2_error_trajectories.mean(axis=0).tolist(),
        'l2_relative_error_std': all_l2_error_trajectories.std(axis=0).tolist(),
        # Interior (MSE do resíduo da EDP)
        'interior_loss_per_run': all_interior_mse_trajectories.tolist(),
        'interior_loss_mean': all_interior_mse_trajectories.mean(axis=0).tolist(),
        'interior_loss_std': all_interior_mse_trajectories.std(axis=0).tolist(),
        # Condições de contorno (MSE)
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

    nome_arquivo_curva = output_dir / f'curva_treino_svpinn_2d_a{a}_{arch_str}.json'
    with open(nome_arquivo_curva, 'w', encoding='utf-8') as f:
        json.dump(training_curve, f, indent=4)

    print(f"Dados salvos como: {nome_arquivo}, {nome_arquivo_pontos} e {nome_arquivo_curva}\n")
