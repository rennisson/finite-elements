from dolfinx import fem, mesh, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import json
import numpy as np
import time
import ufl
import os

PI  = ufl.pi
sin = ufl.sin

def f(x):
    """Define a função RHS: f(x,y,z) = -3*pi^2 * sin(pi*x)*sin(pi*y)*sin(pi*z)."""
    return -3 * PI**2 * sin(PI * x[0]) * sin(PI * x[1]) * sin(PI * x[2])

def u_exact_np(x):
    """Função exata usando NumPy para a interpolação."""
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])


# --- 1. Carregar Pontos de Avaliação (Ground Truth) ---
print("Carregando pontos de avaliação (Ground Truth)...")
gt_file = "gt_poisson_3d.json"
if not os.path.exists(gt_file):
    # Fallback se a malha exata for de 32 ou outro tamanho
    gt_file = "gt_poisson_3d.json"

with open(gt_file, "r") as f_in:
    gt_data = json.load(f_in)

# Achatar as matrizes 3D para listas 1D para avaliação pontual
X_flat = np.array(gt_data["X"]).flatten()
Y_flat = np.array(gt_data["Y"]).flatten()
Z_flat = np.array(gt_data["Z"]).flatten()
U_true = np.array(gt_data["U_true"]).flatten()

# FEniCSx avalia pontos no espaço 3D (x, y, z)
eval_points = np.vstack((X_flat, Y_flat, Z_flat)).T
y_true_norm = np.linalg.norm(U_true)

# Conforme o artigo, N ∈ {16, 32, 64, 128}
N_list = [16, 32, 64, 128]
resultados_fem = {"N": [], "rel_error": [], "solve_time": [], "eval_time": []}

for N in N_list:
    print(f"\n--- Resolvendo FEM 3D para N = {N}x{N}x{N} ---")
    
    start = time.perf_counter()
    msh = mesh.create_box(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), 
        n=(N, N, N),
        cell_type=mesh.CellType.tetrahedron
    )
    end = time.perf_counter()
    print(f"Criação do mesh grid: {end - start:0.6f} segundos")

    V = fem.functionspace(msh, ("Lagrange", 1)) 
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Dirichlet Boundary Condition
    facets = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=lambda coords: np.ones(coords.shape[1], dtype=bool)  # Finds all boundary facets
    )

    dofs_boundary = fem.locate_dofs_topological(
        V=V,
        entity_dim=(msh.topology.dim - 1), 
        entities=facets
    )

    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs_boundary, V=V)
    boundary_conditions = [bc]

    dx = ufl.Measure("cell", domain=msh)
    a  = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L  = ufl.inner(f(x), v) * dx

    # --- Configuração Otimizada de Avaliação (Busca de Células) ---
    # Encontramos de antemão quais células da malha contêm os nossos eval_points.
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

    # --- Loop de Solução e Avaliação 10 vezes (Conforme poisson_2d.py) ---
    for _ in range(10):
        problem = LinearProblem(
            -a,
            L,
            bcs=boundary_conditions,
            petsc_options_prefix="demo_poisson_3d_",
            petsc_options={
                "ksp_type": "cg",
                "ksp_rtol": 1e-6,
                "pc_type": "gamg",
                "pc_gamg_type": "agg",
                "pc_gamg_agg_nsmooths": 1,
                "mg_levels_ksp_type": "chebyshev",
                "mg_levels_pc_type": "jacobi"
            }
        )

        # Tempo de Solução
        t0 = time.time()
        solution = problem.solve()
        tot_solve += (time.time() - t0)

        # Tempo de Avaliação (Interpolate)
        t0_eval = time.time()
        y_approx_raw = solution.eval(valid_points, cells_for_eval)
        tot_eval += (time.time() - t0_eval)

    # Médias dos tempos
    solve_time = tot_solve / 10.0
    eval_time = tot_eval / 10.0

    # --- Cálculo do Erro L2 Relativo Discreto ---
    y_approx = np.zeros_like(U_true)
    y_approx[true_indices] = y_approx_raw.flatten()

    l2 = np.linalg.norm(U_true - y_approx)
    rel_error = l2 / y_true_norm

    print(f"Tempo Médio Solução: {solve_time:.4f} s | Tempo Médio Avaliação: {eval_time:.4f} s")
    print(f"Erro L2 Relativo Discreto: {rel_error:.6e}")

    # --- Salvar resultados na memória ---
    resultados_fem["N"].append(N)
    resultados_fem["rel_error"].append(rel_error)
    resultados_fem["solve_time"].append(solve_time)
    resultados_fem["eval_time"].append(eval_time)


# --- Exportar resultados para JSON ---
save_dir = "fem_poisson_3d"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(save_dir, "fem_results_3d.json"), "w") as f_out:
    json.dump(resultados_fem, f_out, indent=4) 
print(f"\nResultados do FEM 3D salvos com sucesso em '{os.path.join(save_dir, 'fem_results_3d.json')}'!")