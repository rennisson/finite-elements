from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_graphs(
        mesh_domain, u_solutions, times, l2_errors_rel, f_interp_gt,
        t_lower=0, t_upper=1, y_lower=0, y_upper=1
    ) -> None:

    print("\nGerando gráficos...")
        
        # Criar diretório de saída
    Path("results").mkdir(exist_ok=True)
        
        # ===== PLOT 1: Erro L2 Relativo vs Tempo =====
    fig, ax = plt.subplots(figsize=(10, 6))
        
    ax.plot(
        times, l2_errors_rel, 'o-', linewidth=2.5, markersize=7, 
        color='#FF6B6B', markerfacecolor='#FF6B6B', markeredgecolor='#C92A2A',
        markeredgewidth=1.5, label='FEM'
    )
        
    ax.set_xlabel('Erro L2 Relativo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tempo (s)', fontsize=12, fontweight='bold')
    ax.set_title('Allen-Cahn 1D: Erro L2 Relativo vs Tempo', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.6, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    ax.set_yscale('log')
        
    plt.tight_layout()
        
    filepath1 = Path("results") / f"l2_error_vs_time.png"
    plt.savefig(filepath1, dpi=150, bbox_inches='tight')
    print(f"Plot salvo em: {filepath1}")
    plt.close()
        
    # ===== PLOT 2: Comparação FEM vs GT =====
    x_coords = mesh_domain.geometry.x[:, 0]
    sort_idx = np.argsort(x_coords)
        
    u_num = np.array(u_solutions)  # Shape: (num_steps+1, num_dofs)
    u_num_sorted = u_num[:, sort_idx]  # Shape: (num_steps+1, num_spatial_points)
        
    # Interpolar GT para todos os tempos
    u_gt_aligned = f_interp_gt(times)
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
    # Meshgrid para contourf
    time_grid, space_grid = np.meshgrid(times, x_coords[sort_idx])
        
    # FEM - transpor de (temporal, spatial) para (spatial, temporal)
    im1 = axes[0].contourf(time_grid, space_grid, u_num_sorted.T, levels=50, cmap='viridis')
    axes[0].set_xlabel('Tempo', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Espaço', fontsize=11, fontweight='bold')
    axes[0].set_title('Solução FEM', fontsize=12, fontweight='bold')
    axes[0].set_xlim(t_lower, t_upper)
    axes[0].set_ylim(y_lower, y_upper)
    fig.colorbar(im1, ax=axes[0], label='u(x,t)')
        
    # GT - transpor de (temporal, spatial) para (spatial, temporal)
    im2 = axes[1].contourf(time_grid, space_grid, u_gt_aligned.T, levels=50, cmap='viridis')
    axes[1].set_xlabel('Tempo', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Espaço', fontsize=11, fontweight='bold')
    axes[1].set_title('Solução Ground Truth', fontsize=12, fontweight='bold')
    axes[0].set_xlim(t_lower, t_upper)
    axes[0].set_ylim(y_lower, y_upper)
    fig.colorbar(im2, ax=axes[1], label='u(x,t)')
        
    # Erro absoluto
    error_map = np.abs(u_num_sorted.T - u_gt_aligned.T)
    im3 = axes[2].contourf(time_grid, space_grid, error_map, levels=50, cmap='hot')
    axes[2].set_xlabel('Tempo', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Espaço', fontsize=11, fontweight='bold')
    axes[2].set_title('Erro Absoluto |FEM - GT|', fontsize=12, fontweight='bold')
    axes[0].set_xlim(t_lower, t_upper)
    axes[0].set_ylim(y_lower, y_upper)
    fig.colorbar(im3, ax=axes[2], label='|Erro|')
        
    plt.tight_layout()
        
    filepath2 = Path("results") / f"fem_vs_gt_comparacao.png"
    plt.savefig(filepath2, dpi=150, bbox_inches='tight')
    print(f"Plot salvo em: {filepath2}")
    plt.close()