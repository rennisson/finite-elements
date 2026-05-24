from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

folder = "results"

def plot_graphs(
        mesh_sizes, times_list, l2_errors_list, solve_times, 
        u_solutions_list, x_coords_list, f_interp_gt
    ) -> None:

    print("\nGerando gráficos combinados...")
        
    # Criar diretório de saída
    Path(folder).mkdir(exist_ok=True)
    
    # Paleta de cores e marcadores para diferenciar as malhas
    colors = ['#FF6B6B', '#339AF0', '#51CF66', '#FCC419', '#CC5DE8', '#845EF7']
        
    # ===== PLOT 1: Erro L2 vs Tempo de SOLUÇÃO =====
    if len(solve_times) > 0:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plotar cada ponto e conectar com uma linha de tendência (Pareto)
        final_errors = [err[-1] for err in l2_errors_list]
        
        # Linha conectando os pontos
        ax1.plot(final_errors, solve_times, '-', color='gray', alpha=0.4, zorder=1)
        
        for i, m_size in enumerate(mesh_sizes):
            
            ax1.plot(
                final_errors[i], solve_times[i], marker='o', markersize=10,
                color=colors[i], markeredgecolor='black', markeredgewidth=1.2, 
                label=f'FEM (N={m_size})', linestyle='None'
            )
            
        ax1.set_xlabel('Erro L2 Relativo', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Tempo de Solução (s)', fontsize=13, fontweight='bold')
        ax1.set_title('Allen-Cahn 1D: Tempo de Solução vs Erro L² Relativo', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.legend(fontsize=12, loc='best', framealpha=0.9)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        plt.tight_layout()
        filepath_solve = Path(folder) / "l2_error_vs_solve_time_combined.png"
        fig1.savefig(filepath_solve, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"Plot (Erro vs Tempo) salvo em: {filepath_solve}")
        
    # ===== PLOT 2: Estabilidade Numérica - Erro L2 Relativo ao longo do Tempo =====
    fig2, ax2 = plt.subplots(figsize=(10, 6))
        
    for i, m_size in enumerate(mesh_sizes):
        c = colors[i % len(colors)]
        
        # Usamos apenas linha e marcadores pequenos para não poluir o gráfico
        ax2.plot(
            times_list[i], l2_errors_list[i],
            linestyle='--', linewidth=1.0,
            marker='o', markevery=0.05, 
            color=c, 
            label=f'FEM (Nx={m_size})'
        )
        
    ax2.set_xlabel('Tempo (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Erro L2 Relativo', fontsize=12, fontweight='bold')
    ax2.set_title('Allen-Cahn 1D: Estabilidade Numérica - Erro L2 vs Tempo', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.6, linestyle='--')
    ax2.legend(fontsize=11, loc='best')
    ax2.set_yscale('log')
        
    plt.tight_layout()
    filepath_stab = Path("results") / "l2_error_stability_vs_time_combined.png"
    fig2.savefig(filepath_stab, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Plot (Estabilidade) salvo em: {filepath_stab}")

    if u_solutions_list is not None and f_interp_gt is not None:
        for i, m_size in enumerate(mesh_sizes):
            times = np.array(times_list[i])
            x_coords = np.array(x_coords_list[i])
            u_num_sorted = np.array(u_solutions_list[i])  # Shape (Time, Space)
            
            # Preparar a malha GT para interpolar e bater as dimensões
            u_gt_t_raw = f_interp_gt(times)
            x_gt = np.linspace(0, 1.0, u_gt_t_raw.shape[1])
            u_gt_aligned = np.zeros_like(u_num_sorted)
            for t_idx in range(len(times)):
                u_gt_aligned[t_idx, :] = np.interp(x_coords, x_gt, u_gt_t_raw[t_idx, :])

            # Meshgrid para contourf
            time_grid, space_grid = np.meshgrid(times, x_coords)
            
            t_lower, t_upper = times[0], times[-1]
            y_lower, y_upper = x_coords[0], x_coords[-1]

            # Apenas 2 subplots (ignorar erro absoluto por enquanto)
            fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
                
            # FEM - transpor de (temporal, spatial) para (spatial, temporal)
            im1 = axes[0].contourf(time_grid, space_grid, u_num_sorted.T, levels=50, cmap='viridis')
            axes[0].set_xlabel('Tempo', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Espaço', fontsize=11, fontweight='bold')
            axes[0].set_title(f'Solução FEM (Nx={m_size})', fontsize=12, fontweight='bold')
            axes[0].set_xlim(t_lower, t_upper)
            axes[0].set_ylim(y_lower, y_upper)
            fig3.colorbar(im1, ax=axes[0], label='u(x,t)')
                
            # GT - transpor de (temporal, spatial) para (spatial, temporal)
            im2 = axes[1].contourf(time_grid, space_grid, u_gt_aligned.T, levels=50, cmap='viridis')
            axes[1].set_xlabel('Tempo', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Espaço', fontsize=11, fontweight='bold')
            axes[1].set_title('Solução Ground Truth', fontsize=12, fontweight='bold')
            axes[1].set_xlim(t_lower, t_upper)  # <-- Corrigido para axes[1]
            axes[1].set_ylim(y_lower, y_upper)  # <-- Corrigido para axes[1]
            fig3.colorbar(im2, ax=axes[1], label='u(x,t)')
                
            plt.tight_layout()
                
            filepath3 = Path("results") / f"fem_vs_gt_comparacao_Nx{m_size}.png"
            fig3.savefig(filepath3, dpi=150, bbox_inches='tight')
            plt.close(fig3)
            print(f"Plot Contour salvo em: {filepath3}")