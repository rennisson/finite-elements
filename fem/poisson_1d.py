import json
import time
import numpy as np
import ufl
import os

from dolfinx import fem, mesh, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
from pathlib import Path

from fem_plots import plot_graphs_1d

def f(x):
    """Define a função fonte (4x^3 - 6x)e^{-x^2}."""
    return (4*x[0]**3 - 6*x[0]) * (ufl.exp(-x[0]**2))

def left_boundary(x):
    """Fronteira esquerda (condição de Dirichlet)"""
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    """Fronteira direita (condição de Dirichlet)""" 
    return np.isclose(x[0], 1.0)


def solve_poisson_1d(nx: int, x_target: np.ndarray, y_target_exact: np.ndarray):
    """
    Resolve a equação de Poisson 1D usando FEniCSx e avalia a solução
    nos pontos de destino (target) para calcular o erro L2 relativo.
    """
    print(f"\n--- Resolvendo para malha com nx = {nx} ---")
    
    # 1. Criação da Malha e Espaço de Funções
    msh = mesh.create_unit_interval(comm=MPI.COMM_WORLD, nx=nx)
    V = fem.functionspace(msh, ("Lagrange", 1))
    
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # 2. Condições de Contorno de Dirichlet
    facets_left = mesh.locate_entities_boundary(msh, dim=0, marker=left_boundary)
    dofs_left = fem.locate_dofs_topological(V, entity_dim=0, entities=facets_left)
    bc_left = fem.dirichletbc(value=ScalarType(0), dofs=dofs_left, V=V)

    facets_right = mesh.locate_entities_boundary(msh, dim=0, marker=right_boundary)
    dofs_right = fem.locate_dofs_topological(V, entity_dim=0, entities=facets_right)
    bc_right = fem.dirichletbc(value=ScalarType(np.exp(-1.0)), dofs=dofs_right, V=V)

    bcs = [bc_left, bc_right]

    # 3. Formulação Variacional
    dx = ufl.Measure("cell", domain=msh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(f(x), v) * dx

    # 4. Solução do Sistema Linear (Medindo o Solve Time)
    problem = LinearProblem(
        -a, L, bcs=bcs,
        petsc_options_prefix="demo_poisson_1d_",
        petsc_options={"ksp_type": "cg", "pc_type": "ilu"}
    )
    
    t0_solve = time.perf_counter()
    solution = problem.solve()
    solve_time = time.perf_counter() - t0_solve
    
    # 5. AVALIAÇÃO NOS 512 PONTOS DO GROUND TRUTH (Medindo o Eval Time)
    num_points = x_target.shape[0]
    pontos_3d = np.zeros((num_points, 3))
    pontos_3d[:, 0] = x_target.flatten() # O FEniCSx exige coordenadas [x, y, z]

    t0_eval = time.perf_counter()
    
    # Encontrar quais células da malha contêm os pontos do Ground Truth
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pontos_3d)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pontos_3d)

    cells = []
    points_on_proc = []
    exact_filtered = []

    for i, point in enumerate(pontos_3d):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
            exact_filtered.append(y_target_exact[i])

    # Interpolação/Avaliação dos valores do FEM nesses pontos específicos
    y_fem_eval = solution.eval(points_on_proc, cells).flatten()
    
    eval_time = time.perf_counter() - t0_eval
    
    # 6. CÁLCULO DO ERRO L2 RELATIVO DISCRETO
    y_exact_filtered = np.array(exact_filtered).flatten()
    
    norma_erro = np.linalg.norm(y_fem_eval - y_exact_filtered)
    norma_exata = np.linalg.norm(y_exact_filtered)
    rel_error = float(norma_erro / norma_exata)
    
    print(f"Solving Time: {solve_time:.6f} s | Eval Time: {eval_time:.6f} s | Erro L2 Relativo: {rel_error:.6e}")
    
    # Extração de pontos ordenados apenas para o plot individual antigo, se necessário
    points_x = V.tabulate_dof_coordinates()[:, 0] 
    valores_y = solution.x.array
    indices_ordenados = np.argsort(points_x)
    x_ordenado = points_x[indices_ordenados]
    y_ordenado = valores_y[indices_ordenados]
    
    # Opcional: Desative se não quiser que abra vários plots durante o loop
    plot_graphs_1d(mesh_domain=msh, u_exact=lambda x: x*np.exp(-x**2), solution=solution)

    return x_ordenado.reshape(-1, 1), y_ordenado.reshape(-1, 1), solve_time, eval_time, rel_error


def main():
    # 1. Carregar o arquivo Ground Truth gerado anteriormente
    gt_file = "gt_poisson_1d_512.npz"
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"O arquivo '{gt_file}' não foi encontrado. Execute o script 'generate_ground_truth.py' primeiro!")
    
    gt_data = np.load(gt_file)
    x_eval = gt_data["x_eval"]
    u_eval = gt_data["u_eval"]
    
    output_dir = Path("fem_poisson_1d")
    output_dir.mkdir(exist_ok=True)

    mesh_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    # Inicializa o dicionário de resultados incluindo o eval_time
    resultados_fem = {"N": [], "rel_error": [], "solve_time": [], "eval_time": []}
    
    for nx in mesh_sizes:
        x, y, solve_time, eval_time, rel_error = solve_poisson_1d(nx, x_eval, u_eval)
        
        # Popula o dicionário JSON para o gráfico comparativo
        resultados_fem["N"].append(nx)
        resultados_fem["rel_error"].append(rel_error)
        resultados_fem["solve_time"].append(solve_time)
        resultados_fem["eval_time"].append(eval_time)

        # Salva o resultado geométrico NPZ localmente
        nome_arquivo = output_dir / f'dados_fem_nx{nx}.npz'
        np.savez_compressed(
            nome_arquivo, 
            x_fem=x, 
            y_fem=y
        )
    
    # Exporta os resultados globais compilados com a estrutura correta para o plot_comparativo
    with open("fem_poisson_1d/fem_results_1d.json", "w") as f_out:
        json.dump(resultados_fem, f_out, indent=4)
        
    print("\nProcesso concluído!")
    print("Métricas salvas com sucesso em 'fem_poisson_1d/fem_results_1d.json'")

if __name__ == "__main__":
    main()