from plot import plot_graphs

from dolfinx import default_real_type, fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from pathlib import Path
from scipy.interpolate import interp1d

import json
import numpy as np
import os
import ufl


comm      = MPI.COMM_WORLD
pi        = ufl.pi
sin       = ufl.sin
epsilon   = 0.01
T         = 0.05
dt        = 1e-3
num_steps = int(T / dt)
dx        = ufl.dx
mesh_size = [7993]


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
    for msh_size in mesh_size:
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
            petsc_options_prefix="demo_allen_cahn_",
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
    
        # Loop temporal
        t = 0.0
        print("Iniciando simulação com cálculo de erro L2...")
        
        # Carregar GT antes do loop
        has_gt = False
        u_gt = None
        x_coords_num = msh.geometry.x[:, 0]
        sort_idx = np.argsort(x_coords_num)
        
        if os.path.exists('eval_solution_mat.json'):
            with open('eval_solution_mat.json', 'r') as f:
                u_gt = np.array(json.load(f))
            
            # Alinhamento de eixos
            if u_gt.shape[1] != msh_size + 1:
                u_gt = u_gt.T
            
            has_gt = True
            print("Ground Truth carregado!")
            print(f"Shape GT: {u_gt.shape}")
            
            # Preparar interpolador do GT
            t_gt = np.linspace(0, T, u_gt.shape[0])
            f_interp_gt = interp1d(t_gt, u_gt, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        for n in range(num_steps):
            t += dt
    
            _ = problem.solve()
    
            converged_reason = problem.solver.getConvergedReason()
            assert converged_reason > 0
            num_iterations = problem.solver.getIterationNumber()
            print(f"Step {n}: {num_iterations=}")
    
            uh.x.scatter_forward()
    
            for dof_l, dof_r in zip(dofs_left, dofs_right):
                uh.x.array[dof_r] = uh.x.array[dof_l]
    
            u_n.x.array[:] = uh.x.array
    
            u_solutions.append(u_n.x.array.copy())
            times.append(t)
    
            # ===== CÁLCULO DE ERRO L2 RELATIVO POR TIMESTEP =====
            if has_gt:
                u_num_t = u_n.x.array[sort_idx]  # Ordenar espacialmente
                u_gt_t = f_interp_gt(t)  # Interpolar GT no tempo t
                
                diff = u_num_t - u_gt_t
                l2_error_rel = np.linalg.norm(diff) / np.linalg.norm(u_gt_t)
                l2_errors_rel.append(l2_error_rel)
    
        print("Simulação concluída!")
        
        # ===== PLOTS DO ERRO L2 RELATIVO =====
        if has_gt and len(l2_errors_rel) > 1:
            plot_graphs(
                mesh_domain=msh,
                u_solutions=u_solutions, times=times, l2_errors_rel=l2_errors_rel,
                f_interp_gt=f_interp_gt,
                t_upper=T
            )
            
            # ===== ESTATÍSTICAS =====
            print("\n" + "="*55)
            print("ESTATÍSTICAS DO ERRO L2 RELATIVO")
            print("="*55)
            print(f"Erro Mínimo:   {np.min(l2_errors_rel):.6e}")
            print(f"Erro Máximo:   {np.max(l2_errors_rel):.6e}")
            print(f"Erro Médio:    {np.mean(l2_errors_rel):.6e}")
            print(f"Erro Final:    {l2_errors_rel[-1]:.6e}")
            print("="*55)
        else:
            if not has_gt:
                print("Ground Truth não encontrado. Pulando cálculo de erro L2.")
        

        # ===== SALVAR ARRAYS PARA PÓS-PROCESSAMENTO =====
        save_results(msh_size, u_solutions, times, l2_errors_rel, has_gt, x_coords_num, sort_idx)

if __name__ == "__main__":
    solve()
