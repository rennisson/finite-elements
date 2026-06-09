from utils.fem_plots import plot_graphs_2d

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import json
import numpy as np
import time
import ufl


def f(x):
    """Define a função $(4x^3 - 6x)e^{-x^2}$."""
    return 2*(x[0]**4 *(3*x[1]-2) + x[0]**3 *(4-6*x[1]) + x[0]**2 *(6*x[1]**3 - 12*x[1]**2 + 9*x[1] - 2) - 6*x[0]*(x[1] - 1)**2*x[1] + (x[1] - 1)**2*x[1])

def u_exact(x):
    return x[0]**2 * (x[0] - 1)**2 * x[1] * (x[1]-1)**2

def conditions_on_y_zero(x):
    """Fronteira esquerda (ou condiçoes iniciais) do problema"""
    return np.isclose(x[1], 0.0)

def y_one(x):
    """Fronteira esquerda (ou condiçoes iniciais) do problema"""
    return np.isclose(x[1], 0.0)

def x_zero(x):
    """Fronteira esquerda (ou condiçoes iniciais) do problema"""
    return np.isclose(x[0], 0.0)

def x_one(x):
    """Fronteira direita do problema""" 
    return np.isclose(x[0], 1.0)

N_list = [100]

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

    # Dirichlet Boundary Condition on y equals zero
    cond_on_y_zero_facets = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=conditions_on_y_zero
    )

    dofs_y_zero = fem.locate_dofs_topological(
        V=V,
        entity_dim=(msh.topology.dim - 1), 
        entities=cond_on_y_zero_facets
    )

    y_zero_bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs_y_zero, V=V)

    boundary_conditions = [y_zero_bc]

    # Neumann conditions (condition on derivatives)
    facets_neumann_left = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=x_zero 
    )

    facets_neumann_right = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=x_one 
    )

    facets_neumann_top = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=y_one 
    )

    neumann_facets = np.concatenate([facets_neumann_left, facets_neumann_right, facets_neumann_top])
    facet_tags = mesh.meshtags(msh, dim=(msh.topology.dim - 1), entities=neumann_facets, values=1)
    ds = ufl.Measure("exterior_facet", domain=msh, subdomain_data=facet_tags)

    dx = ufl.Measure("cell", domain=msh)
    a  = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L  = ufl.inner(f(x), v) * dx + ufl.inner(ScalarType(0), v) * ds(1)


    problem = LinearProblem(
        -a,
        L,
        bcs=boundary_conditions,
        petsc_options_prefix="demo_poisson_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True}
    )


    t0 = time.time()
    solution = problem.solve()
    solve_time = time.time() - t0

    # 5. Cálculo do Erro L2 Relativo
    # Usamos um espaço de ordem maior para interpolar a solução exata com alta precisão
    V_ex = fem.functionspace(msh, ("Lagrange", 3))
    u_ex = fem.Function(V_ex)
    u_ex.interpolate(u_exact)

    # Formulações do erro
    error_L2_sq = fem.form(ufl.inner(solution - u_ex, solution - u_ex) * dx)
    norm_ex_L2_sq = fem.form(ufl.inner(u_ex, u_ex) * dx)

    # Computar integrais globalmente (útil se rodar em paralelo com MPI)
    error_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_L2_sq), op=MPI.SUM))
    norm_ex = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(norm_ex_L2_sq), op=MPI.SUM))
    
    rel_error = error_L2 / norm_ex

    print(f"Tempo de Solução: {solve_time:.4f} s | Erro L2 Relativo: {rel_error:.6e}")

    # # --- Cálculo do FEM Evaluation Time ---
    # # O artigo cita avaliação em uma malha de 2000 x 2000 células.
    # N_eval = 2000
    # print(f"Avaliando (interpolando) na nova malha de {N_eval}x{N_eval}...")
    
    # msh_eval = mesh.create_rectangle(
    #     comm=MPI.COMM_WORLD,
    #     points=((0.0, 0.0), (1.0, 1.0)), 
    #     n=(N_eval, N_eval),
    #     cell_type=mesh.CellType.triangle
    # )
    # V_eval = fem.functionspace(msh_eval, ("Lagrange", 1))
    # u_eval = fem.Function(V_eval)

    # t0_eval = time.perf_counter()
    # try:
    #     # Transferência entre malhas não coincidentes (dolfinx >= 0.6.0)
    #     nmm_data = fem.create_nonmatching_meshes_interpolation_data(
    #         msh_eval._cpp_object,
    #         V_eval.element,
    #         msh._cpp_object
    #     )
    #     u_eval.interpolate(solution, nmm_interpolation_data=nmm_data)
    # except Exception:
    #     # Fallback para versões anteriores do dolfinx
    #     u_eval.interpolate(solution)
        
    # eval_time = time.perf_counter() - t0_eval
    # print(f"FEM Evaluation Time: {eval_time:0.4f} s")

    # Salvar resultados
    resultados_fem["N"].append(N)
    resultados_fem["rel_error"].append(rel_error)
    resultados_fem["solve_time"].append(solve_time)
    # resultados_fem["eval_time"].append(eval_time)

# --- Exportar resultados para JSON ---
with open("fem_results_2d.json", "w") as f_out:
    json.dump(resultados_fem, f_out, indent=4)
print("\nResultados do FEM 2D salvos em 'fem_results_2d.json'!")

plot_graphs_2d(mesh_domain=msh, u_exact=u_exact, solution=solution)
