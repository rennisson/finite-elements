from dolfinx import default_real_type, fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from pathlib import Path
from scipy.interpolate import interp1d

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import ufl


def initial_condition(x):
    return 0.5 * (0.5 * np.sin(2 * np.pi * x[0]) + 0.5 * np.sin(16 * np.pi * x[0])) + 0.5

comm      = MPI.COMM_WORLD
pi        = ufl.pi
sin       = ufl.sin
epsilon   = 0.01
T         = 0.05
dt        = 1e-3
num_steps = int(T / dt)
dx        = ufl.dx
mesh_size = [7993]

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
    
    # Adicionar condição inicial
    u_solutions.append(u_n.x.array.copy())
    times.append(0.0)

    # Loop temporal
    t = 0.0
    print("Iniciando simulação...")
    for n in range(num_steps):
        t += dt

        _ = problem.solve()

        converged_reason = problem.solver.getConvergedReason()
        assert converged_reason > 0
        num_iterations = problem.solver.getIterationNumber()
        print(f"Step {n}: {converged_reason=} | {num_iterations=}")

        uh.x.scatter_forward()

        for dof_l, dof_r in zip(dofs_left, dofs_right):
            uh.x.array[dof_r] = uh.x.array[dof_l]

        u_n.x.array[:] = uh.x.array

        u_solutions.append(u_n.x.array.copy())
        times.append(t)

    print("Simulação concluída!")

    print("\nCalculando erro em relação ao Ground Truth...")
    u_num = np.array(u_solutions) # Formato inicial FEniCS (51, 7994)
    x_coords_num = msh.geometry.x[:, 0]
    
    # Organizar os Graus de Liberdade (DoFs) espacialmente
    sort_idx = np.argsort(x_coords_num)
    u_num_sorted = u_num[:, sort_idx]

    has_gt = False
    if os.path.exists('eval_solution_mat.json'):
        with open('eval_solution_mat.json', 'r') as f:
            u_gt = np.array(json.load(f))
            
        # Alinhamento de eixos (garantir Tempo x Espaço)
        if u_gt.shape[1] != u_num_sorted.shape[1]:
            u_gt = u_gt.T
            
        # Alinhamento da dimensão do Tempo (100 do GT -> 51 do FEniCS)
        t_gt = np.linspace(0, T, u_gt.shape[0])
        t_num = np.array(times)
        
        # Interpola a matriz do GT apenas ao longo do eixo do tempo (axis=0)
        f_interp_time = interp1d(t_gt, u_gt, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
        u_gt_aligned = f_interp_time(t_num)
        
        has_gt = True
        
        # Cálculo de Erro Relativo L2 com matrizes alinhadas (51, 7994)
        diff = u_num_sorted - u_gt_aligned
        l2_error = np.linalg.norm(diff) / np.linalg.norm(u_gt_aligned)
        print(f"-> Erro L2 Relativo (Tempo Alinhado): {l2_error:.6e}")
        
        # Substitui a referência u_gt pela alinhada para os plots gerarem corretamente
        u_gt = u_gt_aligned 
        
    else:
        print("-> Arquivo 'eval_solution_mat.json' não encontrado. Pulando cálculo de erro.")


    # Plot
    print("\nGerando gráficos comparativos...")
        
    # Criar diretório de saída se não existir
    Path("out_poisson").mkdir(exist_ok=True)

    plot_data = np.array(u_solutions).T 

    # Criar a figura com 2 gráficos lado a lado
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # ------------------------------------------------------------
    # GRÁFICO 1: O "Erro" do Artigo (Visualmente Invertido)
    # ------------------------------------------------------------
    # O parâmetro extent=[0, T, 0, 1] força os labels do eixo Y irem de 0 a 1.
    im1 = axes[0].imshow(
        plot_data, 
        aspect='auto', 
        cmap='viridis', 
        extent=[0, T, 0, 1],
        origin='lower'
    )
    axes[0].set_xlabel('Time', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Space', fontsize=12, fontweight='bold')
    axes[0].set_title('"Inverted" GT Solution', fontsize=13, fontweight='bold')
    fig.colorbar(im1, ax=axes[0], label='u(x,t)')

    # ------------------------------------------------------------
    # GRÁFICO 2: Mapeamento Matemático Correto (Seu estilo original)
    # ------------------------------------------------------------

    # Coordenadas x da malha
    x_coords = msh.geometry.x[:, 0]
    
    # O contourf plota os dados respeitando os grids Cartesianos.
    time_grid, space_grid = np.meshgrid(times, x_coords)
    contourf_plot = axes[1].contourf(time_grid, space_grid, plot_data, levels=50, cmap='viridis')
    axes[1].set_xlabel('Time', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Space', fontsize=12, fontweight='bold')
    axes[1].set_title('FEM Solution', fontsize=13, fontweight='bold')
    axes[1].set_xlim(0, T)
    axes[1].set_ylim(0, 1)
    fig.colorbar(contourf_plot, ax=axes[1], label='u(x,t)')

    # 3. Erro Absoluto Ponto a Ponto
    error_map = np.abs(diff)
    im2 = axes[2].imshow(error_map.T, aspect='auto', cmap='magma', origin='lower', extent=[0, T, 0, 1])
    axes[2].set_title(f'Erro Absoluto (Erro L2={l2_error:.4e})', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time', fontsize=12)
    fig.colorbar(im2, ax=axes[2], label='|Erro|')

    plt.tight_layout()

    # Save fig
    folder_path = Path(__file__).parent.parent / "out_poisson"
    folder_path.mkdir(parents=True, exist_ok=True)
    filepath = folder_path / f"allen_cahn_comparacao_{msh_size}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Gráfico comparativo salvo em '{filepath}'")
    plt.close()