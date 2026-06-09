from matplotlib.tri import Triangulation
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

folder = "results"


def plot_graphs_3d(mesh_domain, u_exact, solution):
    out_folder = Path(folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Extrair coordenadas e valores
    x_coords = mesh_domain.geometry.x[:, 0]
    y_coords = mesh_domain.geometry.x[:, 1]
    z_coords = mesh_domain.geometry.x[:, 2]

    u_values = solution.x.array.real
    u_exact_values = np.array([u_exact(mesh_domain.geometry.x[i]) for i in range(len(mesh_domain.geometry.x))])

    # Calcular min e max global
    vmin = min(u_values.min(), u_exact_values.min())
    vmax = max(u_values.max(), u_exact_values.max())

    # Fatias em 
    z_slice = 0.5
    tolerance = 0.05
    mask_z = np.isclose(z_coords, z_slice, atol=tolerance)

    x_slice = x_coords[mask_z]
    y_slice = y_coords[mask_z]
    u_slice = u_values[mask_z]
    u_exact_slice = u_exact_values[mask_z]

    # Plot 2D das fatias
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Solução numérica
    scatter1 = ax[0].scatter(x_slice, y_slice, c=u_slice, cmap='viridis', s=50, vmin=vmin, vmax=vmax)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title(f'Numerical solution (z={z_slice})')
    ax[0].set_aspect('equal')

    # Solução analítica
    scatter2 = ax[1].scatter(x_slice, y_slice, c=u_exact_slice, cmap='viridis', s=50, vmin=vmin, vmax=vmax)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title(f'Analytical solution (z={z_slice})')
    ax[1].set_aspect('equal')

    # Colorbar
    cbar = plt.colorbar(scatter2, ax=ax, label=f'u(x, y, {z_slice})', shrink=0.8)
    plt.savefig(out_folder / f"poisson_3d_solution.png", dpi=150)
    plt.close()

    return u_slice, u_exact_slice
