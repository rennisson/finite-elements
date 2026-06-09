from matplotlib.tri import Triangulation
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

folder = "results"


def plot_graphs_2d(mesh_domain, solution, u_exact):
    out_folder = Path(folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    x_coords = mesh_domain.geometry.x[:, 0]
    y_coords = mesh_domain.geometry.x[:, 1]

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    u_values = solution.x.array.real
    cells = mesh_domain.geometry.dofmap
    triangulation = Triangulation(x_coords, y_coords, cells)

    # Numerical solution
    contour_fill = ax[0].tricontourf(triangulation, u_values, levels=100, cmap='viridis')
    contour_lines = ax[0].tricontour(triangulation, u_values, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    ax[0].triplot(triangulation, linewidth=0.3, color='black', alpha=0.2)
    ax[0].clabel(contour_lines, inline=True, fontsize=8)
    # plt.colorbar(contour_fill, ax=ax[0], label='u(x, y)')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Numerical solution (FEM)')
    ax[0].set_aspect('equal')

    # Analytical solution
    x = [x_coords, y_coords]
    contour_fill  = ax[1].tricontourf(triangulation, u_exact(x), levels=100, cmap='viridis')
    contour_lines = ax[1].tricontour(triangulation, u_exact(x), levels=10, colors='black', linewidths=0.5, alpha=0.3)
    ax[1].triplot(triangulation, linewidth=0.3, color='black', alpha=0.2)
    ax[1].clabel(contour_lines, inline=True, fontsize=8)
    # plt.colorbar(contour_fill, ax=ax[1], label='u(x, y)', shrink=0.8)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Analytical solution')
    ax[1].set_aspect('equal')
    plt.savefig(out_folder / "poisson_2d_solution.png", dpi=150)
    plt.close()

