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


def f(x):
    """Define a função fonte (4x^3 - 6x)e^{-x^2}."""
    return (4*x[0]**3 - 6*x[0]) * (ufl.exp(-x[0]**2))

def left_boundary(x):
    """Fronteira esquerda (condição de Dirichlet)"""
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    """Fronteira direita (condição de Dirichlet)""" 
    return np.isclose(x[0], 1.0)

def solve_poisson_1d(nx: int, x_target: np.ndarray, y_target_exact: np.ndarray, num_runs=10):
    """
    Resolve a equação de Poisson 1D usando FEniCSx e avalia a solução
    nos pontos de destino. Realiza múltiplas execuções (num_runs) para
    tirar a média do tempo de solução e de avaliação.
    """
    print(f"\n--- Resolvendo para malha com nx = {nx} (Média de {num_runs} execuções) ---")
    
    # 1. Criação da Malha e Espaço de Funções
    msh = mesh.create_unit_interval(comm=MPI.COMM_WORLD, nx=nx)
    V = fem.functionspace(msh, ("Lagrange", 1))
    
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # 2. Condições de Contorno
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

    # Monta o problema linear (feito uma vez fora do loop de medição)
    problem = LinearProblem(
        -a, L, bcs=bcs,
        petsc_options_prefix="demo_poisson_1d_",
        petsc_options={"ksp_type": "cg", "pc_type": "ilu"}
    )
    
    # Pré-processamento geométrico (para a avaliação não enviesar o loop de tempo)
    num_points = x_target.shape[0]
    pontos_3d = np.zeros((num_points, 3))
    pontos_3d[:, 0] = x_target.flatten()

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

    # ==========================================
    # 4. LOOP DE EXECUÇÃO E MEDIÇÃO
    # ==========================================
    total_solve_time = 0.0
    total_eval_time = 0.0
    
    # Opcionalmente, armazenaremos apenas a última solução para cálculo de erro
    final_fem_eval = None
    
    for _ in range(num_runs):
        # A. Medir Tempo de Solução
        t0_solve = time.perf_counter()
        solution = problem.solve()
        t_solve = time.perf_counter() - t0_solve
        total_solve_time += t_solve
        
        # B. Medir Tempo de Avaliação
        t0_eval = time.perf_counter()
        final_fem_eval = solution.eval(points_on_proc, cells).flatten()
        t_eval = time.perf_counter() - t0_eval
        total_eval_time += t_eval

    # Médias dos tempos
    avg_solve_time = total_solve_time / num_runs
    avg_eval_time = total_eval_time / num_runs

    # ==========================================
    # 5. CÁLCULO DO ERRO L2 RELATIVO DISCRETO
    # ==========================================
    y_exact_filtered = np.array(exact_filtered).flatten()
    
    norma_erro = np.linalg.norm(final_fem_eval - y_exact_filtered)
    norma_exata = np.linalg.norm(y_exact_filtered)
    rel_error = float(norma_erro / norma_exata)
    
    print(f"Avg Solving Time: {avg_solve_time:.6f} s | Avg Eval Time: {avg_eval_time:.6f} s | Erro L2: {rel_error:.6e}")
    
    # Extração de pontos
    points_x = V.tabulate_dof_coordinates()[:, 0] 
    valores_y = solution.x.array
    indices_ordenados = np.argsort(points_x)
    x_ordenado = points_x[indices_ordenados]
    y_ordenado = valores_y[indices_ordenados]
    
    return x_ordenado.reshape(-1, 1), y_ordenado.reshape(-1, 1), avg_solve_time, avg_eval_time, rel_error


def main():
    # Substituído gt_poisson_1d_512.npz por JSON
    gt_file = "gt_poisson_1d.json"
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"O arquivo '{gt_file}' não foi encontrado.")
    
    # Lendo arquivo JSON em vez de NPZ
    with open(gt_file, "r") as f_in:
        gt_data = json.load(f_in)
        
    # Importante: Converter listas JSON para arrays do numpy
    x_eval = np.array(gt_data["x_eval"])
    u_eval = np.array(gt_data["u_eval"])
    
    output_dir = Path("fem_poisson_1d")
    output_dir.mkdir(exist_ok=True)
    
    mesh_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    resultados_fem = {"N": [], "rel_error": [], "solve_time": [], "eval_time": []}
    
    for nx in mesh_sizes:
        # Repassando o arquivo ground_truth para a função
        x, y, solve_time, eval_time, rel_error = solve_poisson_1d(nx, x_eval, u_eval, num_runs=10)
        
        resultados_fem["N"].append(nx)
        resultados_fem["rel_error"].append(rel_error)
        resultados_fem["solve_time"].append(solve_time)
        resultados_fem["eval_time"].append(eval_time)

        # Salvando a saída de cada passo num arquivo JSON, mantendo a estrutura
        nome_arquivo = output_dir / f'dados_fem_nx{nx}.json'
        with open(nome_arquivo, "w") as json_out:
            # arrays do numpy precisam ser convertidos usando .tolist() para a serialização no json
            json.dump({
                "x_fem": x.tolist(), 
                "y_fem": y.tolist()
            }, json_out)
    
    with open("fem_poisson_1d/fem_results_1d.json", "w") as f_out:
        json.dump(resultados_fem, f_out, indent=4)
        
    print("\nMétricas salvas com sucesso em 'fem_poisson_1d/fem_results_1d.json'")

if __name__ == "__main__":
    main()