from basix.ufl import _ElementBase
from datetime import datetime
from matplotlib.tri import Triangulation
from numpy.typing import NDArray
from pathlib import Path

import math
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import numpy as np


def plot_graphs_1d(mesh_domain, solution, u_exact):
    folder = 'results'
    x_coords = mesh_domain.geometry.x[:, 0]
    u_values = solution.x.array

    out_folder = Path(folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Numerical solution 
    ax.plot(x_coords, u_values, 'b-o', linewidth=2, markersize=6, label='Numerical Solution (FEM)')
    # Analytical solution
    ax.plot(x_coords, u_exact(x_coords), 'r--', linewidth=2, label='Analytical solution: u(x) = x*exp(-x^2)')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_folder / "poisson_1d_solution.png", dpi=150)
    plt.close()


def plot_graphs_2d(mesh_domain, solution, u_exact):
    folder = 'results'
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



def plot_graphs_3d(mesh_domain, u_exact, solution):
    folder = 'results'
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




images_folder = Path(__file__).resolve().parent.parent / "images"
images_folder.mkdir(parents=True, exist_ok=True)

def ax_configs(ax) -> None:
    # Set limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    
    # Set 3d plot elevation and orientation
    ax.view_init(elev=45, azim=45)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('phi(x,y)', fontsize=10)


def plot_basis_triangle(nodes, ax) -> None:
    x_nodes = nodes[:, 0]
    y_nodes = nodes[:, 1]
    z_nodes = np.zeros(len(x_nodes))

    # Plot nodes using scatter
    ax.scatter(x_nodes, y_nodes, z_nodes, color='red', s=30)
    
    x_perimeter = nodes[:3, 0]
    y_perimeter = nodes[:3, 1]

    # append 0 into the arrays just to complete the triangle plot
    x_perimeter = np.append(x_perimeter, 0)
    y_perimeter = np.append(y_perimeter, 0)
    z_linha = np.zeros(len(x_perimeter))

    # Plot triangle border
    ax.plot(x_perimeter, y_perimeter, z_linha, color='black', linestyle='-')


def plot_basis_functions(element: _ElementBase, lattice: NDArray, basis_functions_values: NDArray) -> None:
    num_basis_functions = basis_functions_values.shape[1]
    print(f"Number of basis functions: {num_basis_functions}")

    # Get triangle vertexes
    nodes = element._element.points
    print("Triangle nodes:")
    for j, node in enumerate(nodes):
        print(f"Node {j}: ({node[0]}, {node[1]})")

    fig = plt.figure(figsize=(12, 10))

    x_tri = lattice[:, 0]
    y_tri = lattice[:, 1]

    for i in range(num_basis_functions):
        nrows = element.degree
        ncols = math.ceil(num_basis_functions / element.degree)
        ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')

        # Creates triangulation
        triangulation = mtri.Triangulation(x_tri, y_tri)
        z = basis_functions_values[:, i]
        
        # Surface plot
        surf = ax.plot_trisurf(
            x_tri, y_tri, z,
            triangles=triangulation.triangles,
            color="#e98850",
            alpha=0.9,
            antialiased=False
        )

        ax.set_title(f'Basis function phi_{i}', fontsize=12, fontweight='bold')
        ax_configs(ax=ax)
        plot_basis_triangle(nodes=nodes, ax=ax)

    fig.tight_layout(w_pad=4.0, h_pad=1.0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(images_folder / f"simulation_{timestamp}", dpi=150, bbox_inches='tight', pad_inches=0.5)
    plt.show()