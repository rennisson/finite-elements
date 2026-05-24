from plot import plot_graphs_2d

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
import numpy as np
import ufl


def f(x):
    """Define a função $(4x^3 - 6x)e^{-x^2}$."""
    return 2*(x[0]**4 *(3*x[1]-2) + x[0]**3 *(4-6*x[1]) + x[0]**2 *(6*x[1]**3 - 12*x[1]**2 + 9*x[1] - 2) - 6*x[0]*(x[1] - 1)**2*x[1] + (x[1] - 1)**2*x[1])

def u_exact(x):
    return x[0]**2 * (x[0] - 1)**2 * x[1] * (x[1]-1)**2

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
plot_graphs_2d(mesh_domain=msh, u_exact=u_exact, solution=solution)
