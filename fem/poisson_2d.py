from fem_plots import plot_graphs_2d

from dolfinx import fem, mesh, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import json
import numpy as np
import time
import ufl
import os


def f(x):
    """Define a função $(4x^3 - 6x)e^{-x^2}$."""
    return 2*(x[0]**4 *(3*x[1]-2) + x[0]**3 *(4-6*x[1]) + x[0]**2 *(6*x[1]**3 - 12*x[1]**2 + 9*x[1] - 2) - 6*x[0]*(x[1] - 1)**2*x[1] + (x[1] - 1)**2*x[1])

def u_exact(x):
    return x[0]**2 * (x[0] - 1)**2 * x[1] * (x[1]-1)**2

def conditions_on_y_zero(x):
    """Fronteira inferior do problema"""
    return np.isclose(x[1], 0.0)

def y_one(x):
    """Fronteira superior do problema"""
    return np.isclose(x[1], 1.0)

def x_zero(x):
    """Fronteira esquerda do problema"""
    return np.isclose(x[0], 0.0)

def x_one(x):
    """Fronteira direita do problema""" 
    return np.isclose(x[0], 1.0)


# --- 1. Carregar Pontos de Avaliação (Ground Truth) ---
print("Carregando pontos de avaliação (Ground Truth)...")
with open("gt_poisson_2d.json", "r") as f_in:
    gt_data = json.load(f_in)

# Achatar as matrizes 2D para listas 1D para avaliação
X_flat = np.array(gt_data["X"]).flatten()
Y_flat = np.array(gt_data["Y"]).flatten()
U_true = np.array(gt_data["U_true"]).flatten()

# FEniCSx avalia pontos no espaço 3D (x, y, z), por isso preenchemos o Z com zeros
eval_points = np.vstack((X_flat, Y_flat, np.zeros_like(X_flat))).T
y_true_norm = np.linalg.norm(U_true)

N_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
resultados_fem = {"N": [], "rel_error": [], "solve_time": [], "eval_time": []}

for N in N_list:

    print(f"\n--- Resolvendo FEM para N = {N}x{N} ---")
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)), 
        n=(N, N),
        cell_type=mesh.CellType.triangle
    )

    V = fem.functionspace(msh, ("Lagrange", 1)) 
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Dirichlet Boundary Condition
    cond_on_y_zero_facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1), marker=conditions_on_y_zero)
    dofs_y_zero = fem.locate_dofs_topological(V=V, entity_dim=(msh.topology.dim - 1), entities=cond_on_y_zero_facets)
    y_zero_bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs_y_zero, V=V)
    boundary_conditions = [y_zero_bc]

    # Neumann conditions
    facets_neumann_left = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1), marker=x_zero)
    facets_neumann_right = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1), marker=x_one)
    facets_neumann_top = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1), marker=y_one)
    
    neumann_facets = np.concatenate([facets_neumann_left, facets_neumann_right, facets_neumann_top])
    facet_tags = mesh.meshtags(msh, dim=(msh.topology.dim - 1), entities=neumann_facets, values=1)
    ds = ufl.Measure("exterior_facet", domain=msh, subdomain_data=facet_tags)
    dx = ufl.Measure("cell", domain=msh)

    a  = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L  = ufl.inner(f(x), v) * dx + ufl.inner(ScalarType(0), v) * ds(1)

    # --- Configuração Otimizada de Avaliação (Busca de Células) ---
    # Para não penalizar o tempo de avaliação com loops do Python,
    # encontramos de antemão quais células da malha contêm os nossos eval_points.
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, eval_points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, eval_points)
    
    offsets = colliding_cells.offsets
    has_cell = offsets[1:] > offsets[:-1] # Filtra apenas os pontos que estão dentro do domínio da malha
    true_indices = np.where(has_cell)[0]
    first_cell_indices = offsets[:-1][has_cell]
    cells_for_eval = colliding_cells.array[first_cell_indices].astype(np.int32)
    valid_points = eval_points[true_indices]

    tot_solve = 0.0
    tot_eval = 0.0

    # --- 2. Loop de Solução e Avaliação 10 vezes ---
    for _ in range(10):
        problem = LinearProblem(
            -a,
            L,
            bcs=boundary_conditions,
            petsc_options_prefix="demo_poisson_",
            petsc_options={"ksp_type": "cg", "pc_type": "ilu", "ksp_error_if_not_converged": True}
        )

        # Tempo de Solução
        t0 = time.time()
        solution = problem.solve()
        tot_solve += (time.time() - t0)

        # Tempo de Avaliação
        t0_eval = time.time()
        y_approx_raw = solution.eval(valid_points, cells_for_eval)
        tot_eval += (time.time() - t0_eval)

    # Médias dos tempos
    solve_time = tot_solve / 10.0
    eval_time = tot_eval / 10.0

    # --- 3. Cálculo do Erro L2 Relativo Discreto ---
    y_approx = np.zeros_like(U_true)
    y_approx[true_indices] = y_approx_raw.flatten()

    l2 = np.linalg.norm(U_true - y_approx)
    rel_error = l2 / y_true_norm

    print(f"Tempo Médio Solução: {solve_time:.4f} s | Tempo Médio Avaliação: {eval_time:.4f} s")
    print(f"Erro L2 Relativo: {rel_error:.6e}")

    # Salvar resultados
    resultados_fem["N"].append(N)
    resultados_fem["rel_error"].append(rel_error)
    resultados_fem["solve_time"].append(solve_time)
    resultados_fem["eval_time"].append(eval_time)


# --- Exportar resultados para JSON ---
save_dir = "fem_poisson_2d"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(save_dir, "fem_results_2d.json"), "w") as f_out:
    json.dump(resultados_fem, f_out, indent=4)
print("\nResultados do FEM 2D salvos em 'fem_results_2d.json'!")

# Função externa de plot (Certifique-se de que sua função saiba lidar com essa 'solution' isolada)
# plot_graphs_2d(mesh_domain=msh, u_exact=u_exact, solution=solution)