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

from ground_truth_poisson_2d_high_freq import generate_analytical_solution_2d

# Parâmetro de frequência da EDP (Eq. 6.4): a in {1, 10, 25, 50, 75}
a = 1


def f(x, a=a):
    """
    Define a função fonte f(x, y) = -Delta u(x, y) para a equação de Poisson 2D
    de alta frequência (Eq. 6.4):
        -Delta u(x, y) = f(x, y),  (x, y) in (0, 1)^2
        u(x, y) = 0,               (x, y) in dOmega
    onde a solução exata é:
        u(x, y) = sin(pi * x) * sin(pi * y) * [sin(a * (x + y)) + sin(2 * pi * x) + cos(3 * pi * y)]
    """
    sin_px = ufl.sin(np.pi * x[0])
    cos_px = ufl.cos(np.pi * x[0])
    sin_py = ufl.sin(np.pi * x[1])
    cos_py = ufl.cos(np.pi * x[1])

    sin_axy = ufl.sin(a * (x[0] + x[1]))
    cos_axy = ufl.cos(a * (x[0] + x[1]))
    sin_2px = ufl.sin(2.0 * np.pi * x[0])
    cos_2px = ufl.cos(2.0 * np.pi * x[0])
    sin_3py = ufl.sin(3.0 * np.pi * x[1])
    cos_3py = ufl.cos(3.0 * np.pi * x[1])

    pi = np.pi
    pi2 = pi ** 2

    # Delta u = term1 + term2 + term3
    term1 = sin_px * sin_py * (
        -2.0 * (pi2 + a ** 2) * sin_axy
        - 6.0 * pi2 * sin_2px
        - 11.0 * pi2 * cos_3py
    )
    term2 = 2.0 * pi * cos_px * sin_py * (a * cos_axy + 2.0 * pi * cos_2px)
    term3 = 2.0 * pi * sin_px * cos_py * (a * cos_axy - 3.0 * pi * sin_3py)

    # Como a EDP é -Delta u = f, temos f = -(term1 + term2 + term3)
    return -(term1 + term2 + term3)


def on_boundary(x):
    """Toda a fronteira dOmega do quadrado unitário (0, 1)^2 (condição de Dirichlet u = 0)"""
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
    )


def solve_poisson_2d(
    nx: int,
    x_target: np.ndarray,
    y_target: np.ndarray,
    u_target_exact: np.ndarray,
    a_val=a,
    num_runs=10
):
    """
    Resolve a equação de Poisson 2D de alta frequência (Eq. 6.4) usando FEniCSx
    em uma malha nx x nx e avalia a solução nos pontos de destino. Realiza múltiplas
    execuções (num_runs) para tirar a média do tempo de solução e de avaliação.
    """
    ny = nx
    print(f"\n--- Resolvendo para malha {nx}x{ny} | a = {a_val} (Média de {num_runs} execuções) ---")

    # 1. Criação da Malha 2D e Espaço de Funções
    msh = mesh.create_unit_square(comm=MPI.COMM_WORLD, nx=nx, ny=ny)
    V = fem.functionspace(msh, ("Lagrange", 1))

    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # 2. Condições de Contorno Homogêneas em dOmega: u(x, y) = 0
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, dim=fdim, marker=on_boundary)
    boundary_dofs = fem.locate_dofs_topological(V, entity_dim=fdim, entities=boundary_facets)
    bc = fem.dirichletbc(value=ScalarType(0.0), dofs=boundary_dofs, V=V)
    bcs = [bc]

    # 3. Formulação Variacional: -Delta u = f  =>  int grad(u) . grad(v) dx = int f v dx
    dx = ufl.Measure("cell", domain=msh)
    bilinear_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(f(x, a=a_val), v) * dx

    # Monta o problema linear (feito uma vez fora do loop de medição)
    problem = LinearProblem(
        bilinear_form, L, bcs=bcs,
        petsc_options_prefix="demo_poisson_2d_high_freq_",
        petsc_options={"ksp_type": "cg", "pc_type": "ilu"}
    )

    # Pré-processamento geométrico (para a avaliação não enviesar o loop de tempo)
    x_flat = x_target.ravel()
    y_flat = y_target.ravel()
    u_exact_flat = u_target_exact.ravel()
    num_points = x_flat.shape[0]

    pontos_3d = np.zeros((num_points, 3), dtype=np.float64)
    pontos_3d[:, 0] = x_flat
    pontos_3d[:, 1] = y_flat

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
            exact_filtered.append(u_exact_flat[i])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)

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
    u_exact_filtered = np.array(exact_filtered).flatten()

    norma_erro = np.linalg.norm(final_fem_eval - u_exact_filtered)
    norma_exata = np.linalg.norm(u_exact_filtered)
    rel_error = float(norma_erro / norma_exata)

    print(f"Avg Solving Time: {avg_solve_time:.6f} s | Avg Eval Time: {avg_eval_time:.6f} s | Erro L2: {rel_error:.6e}")

    # Extração das coordenadas dos DOFs e valores da solução
    dof_coords = V.tabulate_dof_coordinates()
    points_x = dof_coords[:, 0]
    points_y = dof_coords[:, 1]
    valores_u = solution.x.array

    return (
        points_x.reshape(-1, 1),
        points_y.reshape(-1, 1),
        valores_u.reshape(-1, 1),
        avg_solve_time,
        avg_eval_time,
        rel_error,
    )


def main():
    gt_file = f"gt_poisson_2d_high_freq_a{a}.json"
    if not os.path.exists(gt_file):
        print(f"Arquivo '{gt_file}' não encontrado. Gerando automaticamente...")
        generate_analytical_solution_2d(a=a, filename=gt_file)

    # Lendo arquivo JSON do Ground Truth
    with open(gt_file, "r", encoding="utf-8") as f_in:
        gt_data = json.load(f_in)

    # Converter listas JSON para arrays do numpy
    x_eval = np.array(gt_data["X"])
    y_eval = np.array(gt_data["Y"])
    u_eval = np.array(gt_data["U_true"])

    output_dir = Path(f"fem_poisson_2d_high_freq_a{a}")
    output_dir.mkdir(exist_ok=True)

    mesh_sizes = [16, 32, 64, 128, 256, 512]

    resultados_fem = {
        "a": a,
        "N": [],
        "rel_error": [],
        "solve_time": [],
        "eval_time": []
    }

    for nx in mesh_sizes:
        x_fem, y_fem, u_fem, solve_time, eval_time, rel_error = solve_poisson_2d(
            nx, x_eval, y_eval, u_eval, a_val=a, num_runs=10
        )

        resultados_fem["N"].append(nx)
        resultados_fem["rel_error"].append(rel_error)
        resultados_fem["solve_time"].append(solve_time)
        resultados_fem["eval_time"].append(eval_time)

        # Salvando a saída de cada malha num arquivo JSON
        nome_arquivo = output_dir / f"dados_fem_2d_a{a}_nx{nx}.json"
        with open(nome_arquivo, "w", encoding="utf-8") as json_out:
            json.dump({
                "a": a,
                "nx": nx,
                "ny": nx,
                "x_fem": x_fem.tolist(),
                "y_fem": y_fem.tolist(),
                "u_fem": u_fem.tolist()
            }, json_out)

    results_path = output_dir / f"fem_results_2d_a{a}.json"
    with open(results_path, "w", encoding="utf-8") as f_out:
        json.dump(resultados_fem, f_out, indent=4)

    print(f"\nMétricas salvas com sucesso em '{results_path}'")


if __name__ == "__main__":
    main()
