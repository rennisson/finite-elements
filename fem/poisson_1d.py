import json
import time
import numpy as np
import ufl

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
from pathlib import Path

from fem_plots import plot_graphs_1d

def f(x):
    """Define a função fonte (4x^3 - 6x)e^{-x^2}."""
    return (4*x[0]**3 - 6*x[0]) * (ufl.exp(-x[0]**2))

def u_exact_np(x):
    """Função exata para o cálculo de erro e interpolação do FEniCSx no NumPy."""
    return x[0] * np.exp(-x[0]**2)

def u_exact(x):
    """Mantido para compatibilidade com a função plot_graphs_1d."""
    return x * np.exp(-x**2)

def left_boundary(x):
    """Fronteira esquerda (condição de Dirichlet)"""
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    """Fronteira direita (condição de Dirichlet)""" 
    return np.isclose(x[0], 1.0)


def solve_poisson_1d(nx: int):
    """
    Resolve a equação de Poisson 1D usando FEniCSx para um dado tamanho de malha nx.
    
    Retorna:
        tuple: (x_ordenado, y_ordenado, solve_time, rel_error, eval_time)
    """
    print(f"\n--- Resolvendo para malha com nx = {nx} ---")
    
    # Criação da Malha e Espaço de Funções
    msh = mesh.create_unit_interval(comm=MPI.COMM_WORLD, nx=nx)
    V = fem.functionspace(msh, ("Lagrange", 1))
    
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Condições de Contorno de Dirichlet
    facets_left = mesh.locate_entities_boundary(msh, dim=0, marker=left_boundary)
    dofs_left = fem.locate_dofs_topological(V, entity_dim=0, entities=facets_left)
    bc_left = fem.dirichletbc(value=ScalarType(0), dofs=dofs_left, V=V)

    facets_right = mesh.locate_entities_boundary(msh, dim=0, marker=right_boundary)
    dofs_right = fem.locate_dofs_topological(V, entity_dim=0, entities=facets_right)
    bc_right = fem.dirichletbc(value=ScalarType(np.exp(-1.0)), dofs=dofs_right, V=V)

    bcs = [bc_left, bc_right]

    # Formulação Variacional
    dx = ufl.Measure("cell", domain=msh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(f(x), v) * dx

    # Solução do Sistema Linear
    problem = LinearProblem(
        -a, L, bcs=bcs,
        petsc_options_prefix="demo_poisson_1d_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    
    # --- Medição do Solving Time ---
    t0_solve = time.perf_counter()
    solution = problem.solve()
    solve_time = time.perf_counter() - t0_solve
    
    # --- Cálculo do Erro L2 Relativo ---
    V_ex = fem.functionspace(msh, ("Lagrange", 3))
    u_ex = fem.Function(V_ex)
    u_ex.interpolate(u_exact_np)

    error_L2_sq = fem.form(ufl.inner(solution - u_ex, solution - u_ex) * dx)
    norm_ex_L2_sq = fem.form(ufl.inner(u_ex, u_ex) * dx)

    error_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_L2_sq), op=MPI.SUM))
    norm_ex = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(norm_ex_L2_sq), op=MPI.SUM))
    rel_error = error_L2 / norm_ex
    
    print(f"Solving Time: {solve_time:.6f} s | Erro L2 Relativo: {rel_error:.6e}")
    
    # --- EXTRAÇÃO E ORDENAÇÃO DOS DADOS ---
    points_x = V.tabulate_dof_coordinates()[:, 0] 
    valores_y = solution.x.array
    
    indices_ordenados = np.argsort(points_x)
    x_ordenado = points_x[indices_ordenados]
    y_ordenado = valores_y[indices_ordenados]
    
    plot_graphs_1d(mesh_domain=msh, u_exact=u_exact, solution=solution)

    return x_ordenado.reshape(-1, 1), y_ordenado.reshape(-1, 1), solve_time, rel_error


def main():
    output_dir = Path("fem_poisson_1d")
    output_dir.mkdir(exist_ok=True)
    
    mesh_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    # Inicializa o dicionário de resultados
    resultados_fem = {"N": [], "rel_error": [], "solve_time": []}
    
    for nx in mesh_sizes:
        x, y, solve_time, rel_error = solve_poisson_1d(nx)
        
        # Popula o dicionário JSON
        resultados_fem["N"].append(nx)
        resultados_fem["rel_error"].append(rel_error)
        resultados_fem["solve_time"].append(solve_time)

        # Salva o resultado NPZ localmente
        nome_arquivo = output_dir / f'dados_fem_nx{nx}.npz'
        np.savez_compressed(
            nome_arquivo, 
            x_fem=x, 
            y_fem=y
        )
    
    # Exporta os resultados globais compilados
    with open("fem_poisson_1d/fem_results_1d.json", "w") as f_out:
        json.dump(resultados_fem, f_out, indent=4)
        
    print("\nProcesso concluído! Métricas salvos em 'fem_results_1d.json' e dados geométricos na pasta 'npz_fem'.")

if __name__ == "__main__":
    main()