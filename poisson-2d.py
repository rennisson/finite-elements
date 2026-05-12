from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from matplotlib.tri import Triangulation
from pathlib import Path
from petsc4py.PETSc import ScalarType  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import ufl


def f(x):
    """Define a função $(4x^3 - 6x)e^{-x^2}$."""
    return 2*(x[0]**4 *(3*x[1]-2) + x[0]**3 *(4-6*x[1]) + x[0]**2 *(6*x[1]**3 - 12*x[1]**2 + 9*x[1] - 2) - 6*x[0]*(x[1] - 1)**2*x[1] + (x[1] - 1)**2*x[1])

def conditions_on_y_zero(x):
    """Fronteira esquerda (ou condiçoes iniciais) do problema"""
    return np.isclose(x[1], 0.0)

def y_one(x):
    """Fronteira esquerda (ou condiçoes iniciais) do problema"""
    return np.isclose(x[1], 0.0)

def x_zero(x):
    """Fronteira esquerda (ou condiçoes iniciais) do problema"""
    return np.isclose(x[0], 0.0)

def x_one(x):
    """Fronteira direita do problema""" 
    return np.isclose(x[0], 1.0)

def u_exact(x):
    return x[0]**2 * (x[0] - 1)**2 * x[1] * (x[1]-1)**2


msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)), 
    n=(100, 100),
    cell_type=mesh.CellType.triangle
)

V = fem.functionspace(msh, ("Lagrange", 1)) 
v = ufl.TestFunction(V)
u = ufl.TrialFunction(V)
x = ufl.SpatialCoordinate(msh)

# Dirichlet Boundary Condition on y equals zero
cond_on_y_zero_facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=conditions_on_y_zero
)

dofs_y_zero = fem.locate_dofs_topological(
    V=V,
    entity_dim=(msh.topology.dim - 1), 
    entities=cond_on_y_zero_facets
)

y_zero_bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs_y_zero, V=V)

boundary_conditions = [y_zero_bc]

# Neumann conditions (condition on derivatives)
facets_neumann_left = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=x_zero 
)

facets_neumann_right = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=x_one 
)

facets_neumann_top = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=y_one 
)

neumann_facets = np.concatenate([facets_neumann_left, facets_neumann_right, facets_neumann_top])
facet_tags = mesh.meshtags(msh, dim=(msh.topology.dim - 1), entities=neumann_facets, values=1)
ds = ufl.Measure("exterior_facet", domain=msh, subdomain_data=facet_tags)

dx = ufl.Measure("cell", domain=msh)
a  = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
L  = ufl.inner(f(x), v) * dx + ufl.inner(ScalarType(0), v) * ds(1)


problem = LinearProblem(
    -a,
    L,
    bcs=boundary_conditions,
    petsc_options_prefix="demo_poisson_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True}
)


solution = problem.solve()

out_folder = Path("out_poisson")
out_folder.mkdir(parents=True, exist_ok=True)

x_coords = msh.geometry.x[:, 0]
y_coords = msh.geometry.x[:, 1]

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

u_values = solution.x.array.real
cells = msh.geometry.dofmap
triangulation = Triangulation(x_coords, y_coords, cells)

# Numerical solution
contour_fill = ax[0].tricontourf(triangulation, u_values, levels=100, cmap='viridis')
# contour_lines = ax[0].tricontour(triangulation, u_values, levels=10, colors='black', linewidths=0.5, alpha=0.3)
ax[0].triplot(triangulation, linewidth=0.3, color='black', alpha=0.2)
# ax[0].clabel(contour_lines, inline=True, fontsize=8)
# plt.colorbar(contour_fill, ax=ax[0], label='u(x, y)')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title('Numerical solution (FEM)')
ax[0].set_aspect('equal')

# Analytical solution
x = [x_coords, y_coords]
contour_fill  = ax[1].tricontourf(triangulation, u_exact(x), levels=100, cmap='viridis')
# contour_lines = ax[1].tricontour(triangulation, u_exact(x), levels=10, colors='black', linewidths=0.5, alpha=0.3)
ax[1].triplot(triangulation, linewidth=0.3, color='black', alpha=0.2)
# ax[1].clabel(contour_lines, inline=True, fontsize=8)
# plt.colorbar(contour_fill, ax=ax[1], label='u(x, y)')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title('Analytical solution')
ax[1].set_aspect('equal')
plt.tight_layout()
plt.savefig(out_folder / "poisson_2d_solution.png", dpi=150)
plt.close()
