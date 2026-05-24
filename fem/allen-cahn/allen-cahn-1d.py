from plot import plot_graphs

from dolfinx import default_real_type, fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from pathlib import Path
from scipy.interpolate import interp1d

import json
import numpy as np
import os
import time
import ufl


comm      = MPI.COMM_WORLD
pi        = ufl.pi
sin       = ufl.sin
epsilon   = 0.01
T         = 0.05
dt        = 1e-3
num_steps = int(T / dt)
dx        = ufl.dx
mesh_size = [32, 128, 512, 2048, 7993]


def initial_condition(x):
    return 0.5 * (0.5 * np.sin(2 * np.pi * x[0]) + 0.5 * np.sin(16 * np.pi * x[0])) + 0.5


def save_results(
        msh_size, u_solutions, times, l2_errors_rel, has_gt, x_coords_num, sort_idx
    ):
    print("\nSalvando arrays para pós-processamento...")
    Path("results").mkdir(exist_ok=True)
    
    # Salvar timesteps
    times_array = np.array(times)
    times_file = Path("results") / f"times_{msh_size}.npy"
    np.save(times_file, times_array)
    print(f"Timesteps salvos em: {times_file}")
    
    # Salvar soluções FEM ordenadas espacialmente
    u_fem_array = np.array(u_solutions)[:, sort_idx]  # (num_steps, num_spatial_points)
    u_fem_file = Path("results") / f"u_fem_solutions_{msh_size}.npy"
    np.save(u_fem_file, u_fem_array)
    print(f"Soluções FEM salvas em: {u_fem_file}")
    
    # Salvar erros L2 relativo
    if has_gt and len(l2_errors_rel) > 1:
        l2_errors_array = np.array(l2_errors_rel)
        l2_errors_file = Path("results") / f"l2_errors_{msh_size}.npy"
        np.save(l2_errors_file, l2_errors_array)
        print(f"Erros L2 salvos em: {l2_errors_file}")
    
    # Salvar coordenadas espaciais
    x_coords_file = Path("results") / f"x_coords_{msh_size}.npy"
    np.save(x_coords_file, x_coords_num[sort_idx])
    print(f"Coordenadas espaciais salvas em: {x_coords_file}")
    
    # Salvar metadados em JSON
    metadata = {
        'mesh_size': int(msh_size),
        'epsilon': float(epsilon),
        'T': float(T),
        'dt': float(dt),
        'num_steps': int(num_steps),
        'num_spatial_points': int(u_fem_array.shape[1]),
        'has_ground_truth': bool(has_gt)
    }
    
    metadata_file = Path("results") / f"metadata_{msh_size}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadados salvos em: {metadata_file}")

    return metadata_file


def solve():
    # Estruturas para acumular os dados de todos os mesh_sizes
    all_mesh_sizes = []
    all_times = []
    all_l2_errors = []
    all_solve_times = []
    all_u_solutions = []
    all_x_coords = []
    
    f_interp_gt_global = None

    for msh_size in mesh_size:
        print(f"\n{'='*50}\nIniciando simulação para malha Nx = {msh_size}\n{'='*50}")
        # Meshgrid e Espaço de Funções
        msh = mesh.create_unit_interval(comm, nx=msh_size)
        V = fem.functionspace(msh, ("Lagrange", 1))

        uh = fem.Function(V) # t_(n+1)
        u_n = fem.Function(V) # t_n
        v = ufl.TestFunction(V)

        # Interpola a condição inicial para uh e copia para u_n
        uh.interpolate(initial_condition)
        uh.x.scatter_forward()
        u_n.x.array[:] = uh.x.array[:]

        # Aplica condições de contorno
        dofs_left  = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
        dofs_right = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))

        # Formulação Fraca (Euler Implícito)
        F0 = (uh - u_n) * v * dx
        F1 = dt * epsilon * ufl.dot(ufl.grad(uh), ufl.grad(v)) * dx
        F2 = dt * (2.0 / epsilon) * u_n * (1.0 - u_n) * (1.0 - 2.0 * u_n) * v * dx
        F = F0 + F1 + F2 # Residuo não-linear

        # Problema não linear e Newton Solver
        problem = NonlinearProblem(
            F, uh,
            petsc_options_prefix=f"demo_allen_cahn_{msh_size}_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "none",
                "snes_stol": np.sqrt(np.finfo(default_real_type).eps) * 1e-2,
                "snes_atol": 1e-8,
                "snes_rtol": 1e-8,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "petsc",
                "snes_monitor": None,
            }
        )

        u_solutions = []
        times = []
        l2_errors_rel = []
        
        # Adicionar condição inicial
        u_solutions.append(u_n.x.array.copy())
        times.append(0.0)
        l2_errors_rel.append(0.0)  # Erro inicial é zero
    
        t = 0.0
        
        # Carregar GT antes do loop
        has_gt = False
        u_gt = None
        x_coords_num = msh.geometry.x[:, 0]
        sort_idx = np.argsort(x_coords_num)
        
        if os.path.exists('eval_solution_mat.json'):
            with open('eval_solution_mat.json', 'r') as f:
                u_gt = np.array(json.load(f))
            
            has_gt = True
            
            # Preparar interpolador do GT no tempo
            t_gt = np.linspace(0, T, u_gt.shape[0])
            f_interp_gt = interp1d(t_gt, u_gt, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
            x_gt = np.linspace(0, 1.0, u_gt.shape[1])
        
        time_start = time.time()

        for n in range(num_steps):
            t += dt
            _ = problem.solve()
    
            converged_reason = problem.solver.getConvergedReason()
            assert converged_reason > 0
    
            uh.x.scatter_forward()
    
            for dof_l, dof_r in zip(dofs_left, dofs_right):
                uh.x.array[dof_r] = uh.x.array[dof_l]
    
            u_n.x.array[:] = uh.x.array
    
            u_solutions.append(u_n.x.array.copy())
            times.append(t)
    
            # ===== CÁLCULO DE ERRO L2 RELATIVO =====
            if has_gt:
                u_num_t = u_n.x.array[sort_idx] 
                u_gt_t = f_interp_gt(t)          
                
                x_num = x_coords_num[sort_idx]
                u_gt_t_interp = np.interp(x_num, x_gt, u_gt_t)
                
                diff = u_num_t - u_gt_t_interp
                
                norm_gt = np.linalg.norm(u_gt_t_interp)
                l2_error_rel = np.linalg.norm(diff) / norm_gt if norm_gt > 0 else np.linalg.norm(diff)
                l2_errors_rel.append(l2_error_rel)
        
        time_end = time.time()
        solve_time = time_end - time_start
    
        print(f"Simulação para Nx={msh_size} concluída!")
        
        if has_gt:
            f_interp_gt_global = f_interp_gt # Guarda a referência

        if has_gt and len(l2_errors_rel) > 1:
            all_mesh_sizes.append(msh_size)
            all_times.append(times)
            all_l2_errors.append(l2_errors_rel)
            all_solve_times.append(solve_time)
            
            # Salvar array FEM ordenado espacialmente e as coordenadas ordenadas
            all_u_solutions.append(np.array(u_solutions)[:, sort_idx])
            all_x_coords.append(x_coords_num[sort_idx])
            
            print(f"Erro Final: {l2_errors_rel[-1]:.6e} | Tempo: {solve_time:.4f} s")

        save_results(msh_size, u_solutions, times, l2_errors_rel, has_gt, x_coords_num, sort_idx)


    if len(all_mesh_sizes) > 0:
        plot_graphs(
            mesh_sizes=all_mesh_sizes,
            times_list=all_times,
            l2_errors_list=all_l2_errors,
            solve_times=all_solve_times,
            u_solutions_list=all_u_solutions,
            x_coords_list=all_x_coords,
            f_interp_gt=f_interp_gt_global
        )
    else:
        print("Nenhum dado com Ground Truth pôde ser plotado.")


if __name__ == "__main__":
    solve()
