from basix.ufl import _ElementBase
from datetime import datetime
from numpy.typing import NDArray
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

images_folder = Path(__file__).resolve().parent.parent / "images"
images_folder.mkdir(parents=True, exist_ok=True)


def plot_basis_functions(element: _ElementBase, lattice: NDArray, basis_functions_values: NDArray):
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
        ncols = int(num_basis_functions / element.degree)
        ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')

        # Creates triangulation
        triangulation = mtri.Triangulation(x_tri, y_tri)
        z = basis_functions_values[:, i]
        
        # Plot da superfície
        surf = ax.plot_trisurf(
            x_tri, y_tri, z,
            triangles=triangulation.triangles,
            color="#e98850",
            alpha=0.9,
            antialiased=False
        )
        
        for j, node in enumerate(nodes):
            ax.scatter(node[0], node[1], 0, color='black', marker='o')
        
        nodes_closed = np.vstack((nodes, nodes[0]))

        # Separates x and y
        x = nodes_closed[:, 0]
        y = nodes_closed[:, 1]
        z = np.zeros(len(x)) # Create a zero-array for Z-axis

        ax.plot(x, y, z, color='black')
        
        # Set limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        
        # Set 3d plot elevation and orientation
        ax.view_init(elev=45, azim=45)
        
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_zlabel('phi(x,y)', fontsize=10)
        ax.set_title(f'Basis function phi_{i}', fontsize=12, fontweight='bold')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(images_folder / f"simulacao_{timestamp}", dpi=150, bbox_inches='tight')
    plt.show()