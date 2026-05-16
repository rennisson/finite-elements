import basix
import basix.ufl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

default_real_type = np.float64

element = basix.ufl.element(
    family=basix.ElementFamily.P,
    cell=basix.CellType.triangle,
    degree=1,
    lagrange_variant=basix.LagrangeVariant.equispaced,
    dtype=default_real_type,
)

lattice = basix.create_lattice(basix.CellType.triangle, 50, basix.LatticeType.equispaced, True)

x_tri = lattice[:, 0]
y_tri = lattice[:, 1]
values = element.tabulate(0, lattice)[0, :, :]

num_basis_functions = values.shape[1]

# Obter os nós (vértices do triângulo)
nodes = element._element.points
print("Nós do triângulo:")

fig = plt.figure(figsize=(16, 5))

for i in range(num_basis_functions):
    ax = fig.add_subplot(1, num_basis_functions, i+1, projection='3d')
    
    # Criar triangulação
    triangulation = mtri.Triangulation(x_tri, y_tri)
    
    z = values[:, i]
    
    # Plot da superfície
    surf = ax.plot_trisurf(
        x_tri, 
        y_tri, 
        z,
        triangles=triangulation.triangles,
        cmap='viridis',
        alpha=0.9,
        edgecolor='black',
        linewidth=0.2
    )
    for j, node in enumerate(nodes):
        ax.scatter(node[0], node[1], 0, color='black', marker='o')
    
    # FIXAR OS LIMITES - MESMA ORIENTAÇÃO PARA TODOS
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1.1])
    
    # FIXAR O ÂNGULO DE VISUALIZAÇÃO - MESMA ORIENTAÇÃO
    ax.view_init(elev=25, azim=45)
    
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('φ(x,y)', fontsize=10)
    ax.set_title(f'Função de Base φ_{i}', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(surf, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Valor')

plt.tight_layout()
plt.savefig("lagrange_p1_triangle_3d.png", dpi=150, bbox_inches='tight')
plt.show()