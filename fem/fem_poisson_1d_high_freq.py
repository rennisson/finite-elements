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

from ground_truth_poisson_1d_high_freq import generate_analytical_solution

# Parâmetro de frequência da EDP (Eq. 6.3)
a = 1


def f(x, a=a):
    """Define a função fonte -4*pi^2*sin(2*pi*x) - 0.1*(a*pi)^2*sin(a*pi*x) (Eq. 6.3)."""
    return (
        -4.0 * (np.pi ** 2) * ufl.sin(2.0 * np.pi * x[0])
        - 0.1 * ((a * np.pi) ** 2) * ufl.sin(a * np.pi * x[0])
    )


def left_boundary(x):
    """Fronteira esquerda (condição de Dirichlet u(0) = 0)"""
    return np.isclose(x[0], 0.0)


def right_boundary(x):
    """Fronteira direita (condição de Dirichlet u(1) = 0)"""
    return np.isclose(x[0], 1.0)


def solve_poisson_1d(nx: int, x_target: np.ndarray, y_target_exact: np.ndarray, a_val=a, num_runs=10):
    """
    Resolve a equação de Poisson 1D de alta frequência (Eq. 6.3) usando FEniCSx
    e avalia a solução nos pontos de destino. Realiza múltiplas execuções (num_runs)
    para tirar a média do tempo de solução e de avaliação.
    """
    print(f"\n--- Resolvendo para malha com nx = {nx} | a = {a_val} (Média de {num_runs} execuções) ---")

    # 1. Criação da Malha e Espaço de Funções
    msh = mesh.create_unit_interval(comm=MPI.COMM_WORLD, nx=nx)
    V = fem.functionspace(msh, ("Lagrange", 1))

    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # 2. Condições de Contorno Homogêneas: u(0) = u(1) = 0
    facets_left = mesh.locate_entities_boundary(msh, dim=0, marker=left_boundary)
    dofs_left = fem.locate_dofs_topological(V, entity_dim=0, entities=facets_left)
    bc_left = fem.dirichletbc(value=ScalarType(0.0), dofs=dofs_left, V=V)

    facets_right = mesh.locate_entities_boundary(msh, dim=0, marker=right_boundary)
    dofs_right = fem.locate_dofs_topological(V, entity_dim=0, entities=facets_right)
    bc_right = fem.dirichletbc(value=ScalarType(0.0), dofs=dofs_right, V=V)

    bcs = [bc_left, bc_right]

    # 3. Formulação Variacional: u_xx = f  =>  - int u_x v_x dx = int f v dx
    dx = ufl.Measure("cell", domain=msh)
    bilinear_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(f(x, a=a_val), v) * dx

    # Monta o problema linear (feito uma vez fora do loop de medição)
    problem = LinearProblem(
        -bilinear_form, L, bcs=bcs,
        petsc_options_prefix="demo_poisson_1d_high_freq_",
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
    gt_file = f"gt_poisson_1d_high_freq_a{a}.json"
    if not os.path.exists(gt_file):
        print(f"Arquivo '{gt_file}' não encontrado. Gerando automaticamente...")
        generate_analytical_solution(a=a, filename=gt_file)

    # Lendo arquivo JSON
    with open(gt_file, "r", encoding="utf-8") as f_in:
        gt_data = json.load(f_in)

    # Converter listas JSON para arrays do numpy
    x_eval = np.array(gt_data["X"])
    u_eval = np.array(gt_data["U_true"])

    output_dir = Path(f"fem_poisson_1d_high_freq_a{a}")
    output_dir.mkdir(exist_ok=True)

    mesh_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    resultados_fem = {
        "a": a,
        "N": [],
        "rel_error": [],
        "solve_time": [],
        "eval_time": []
    }

    for nx in mesh_sizes:
        x, y, solve_time, eval_time, rel_error = solve_poisson_1d(
            nx, x_eval, u_eval, a_val=a, num_runs=10
        )

        resultados_fem["N"].append(nx)
        resultados_fem["rel_error"].append(rel_error)
        resultados_fem["solve_time"].append(solve_time)
        resultados_fem["eval_time"].append(eval_time)

        # Salvando a saída de cada malha num arquivo JSON
        nome_arquivo = output_dir / f"dados_fem_a{a}_nx{nx}.json"
        with open(nome_arquivo, "w", encoding="utf-8") as json_out:
            json.dump({
                "a": a,
                "x_fem": x.tolist(),
                "y_fem": y.tolist()
            }, json_out)

    results_path = output_dir / f"fem_results_1d_a{a}.json"
    with open(results_path, "w", encoding="utf-8") as f_out:
        json.dump(resultados_fem, f_out, indent=4)

    print(f"\nMétricas salvas com sucesso em '{results_path}'")


if __name__ == "__main__":
    main()
