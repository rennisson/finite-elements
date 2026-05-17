from basix.ufl import _ElementBase
from datetime import datetime
from numpy.typing import NDArray
from pathlib import Path

import math
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

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