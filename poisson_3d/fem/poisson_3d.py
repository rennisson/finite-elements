from utils.fem_plots import plot_graphs_3d

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import json
import numpy as np
import time
import ufl

PI  = ufl.pi
sin = ufl.sin
N_list = [16]

def f(x):
    """Define a função."""
    return -3 * PI**2 * sin(PI * x[0]) * sin(PI * x[1]) * sin(PI * x[2])

def u_exact_np(x):
    """Função exata usando NumPy para a interpolação (evita conflito com UFL)."""
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])

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

    start = time.perf_counter()
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
    end = time.perf_counter()
    print(f"Definindo e aplicando condições de contorno: {end - start:0.6f} segundos")

    start = time.perf_counter()
    dx = ufl.Measure("cell", domain=msh)
    a  = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L  = ufl.inner(f(x), v) * dx

    problem = LinearProblem(
        -a,
        L,
        bcs=boundary_conditions,
        petsc_options_prefix="demo_poisson_3d_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True}
    )

    # FEM Solving time
    t0 = time.time()
    solution = problem.solve()
    solve_time = time.time() - t0
    print(f"Resolvendo o sistema: {solve_time:0.6f} segundos")

    # --- Cálculo do Erro L2 Relativo ---
    # Usamos um espaço de ordem maior para interpolar a solução exata com alta precisão
    V_ex = fem.functionspace(msh, ("Lagrange", 3))
    u_ex = fem.Function(V_ex)
    u_ex.interpolate(u_exact_np)

    # Formulações do erro
    error_L2_sq = fem.form(ufl.inner(solution - u_ex, solution - u_ex) * dx)
    norm_ex_L2_sq = fem.form(ufl.inner(u_ex, u_ex) * dx)

    # Computar integrais globalmente (compatível com MPI)
    error_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_L2_sq), op=MPI.SUM))
    norm_ex = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(norm_ex_L2_sq), op=MPI.SUM))
    
    rel_error = error_L2 / norm_ex

    print(f"Tempo de Solução: {solve_time:.4f} s | Erro L2 Relativo: {rel_error:.6e}")

    # # --- Cálculo do FEM Evaluation Time ---
    # # De acordo com o trecho do artigo, a solução deve ser avaliada (interpolada) 
    # # em uma nova malha densa de 150x150x150 pontos para comparação com a PINN.
    # N_eval = 150
    # print(f"Avaliando (interpolando) na malha de {N_eval}x{N_eval}x{N_eval}...")
    # msh_eval = mesh.create_box(
    #     comm=MPI.COMM_WORLD,
    #     points=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), 
    #     n=(N_eval, N_eval, N_eval),
    #     cell_type=mesh.CellType.tetrahedron
    # )
    # V_eval = fem.functionspace(msh_eval, ("Lagrange", 1))
    # u_eval = fem.Function(V_eval)

    # t0_eval = time.perf_counter()
    # try:
    #     # Nas versões >=0.6.0 do FEniCSx, é necessário criar os dados de interpolação 
    #     # para transferir a solução entre malhas de resoluções diferentes.
    #     nmm_data = fem.create_nonmatching_meshes_interpolation_data(
    #         msh_eval._cpp_object,
    #         V_eval.element,
    #         msh._cpp_object
    #     )
    #     u_eval.interpolate(solution, nmm_interpolation_data=nmm_data)
    # except Exception:
    #     # Fallback caso utilize uma versão mais antiga onde a função trata isso direto
    #     u_eval.interpolate(solution)
        
    # eval_time = time.perf_counter() - t0_eval
    # print(f"FEM Evaluation Time (Interpolação): {eval_time:0.6f} segundos")

    # --- Salvar resultados na memória ---
    resultados_fem["N"].append(N)
    resultados_fem["rel_error"].append(rel_error)
    resultados_fem["solve_time"].append(solve_time)
    # resultados_fem["eval_time"].append(eval_time)

# --- Exportar resultados para JSON ---
with open("fem_results_3d.json", "w") as f_out:
    # indent=4 deixa o arquivo estruturado e fácil de ler
    json.dump(resultados_fem, f_out, indent=4) 
print("\nResultados do FEM 3D salvos com sucesso em 'fem_results_3d.json'!")

plot_graphs_3d(mesh_domain=msh, u_exact=u_exact_np, solution=solution)