from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from pathlib import Path
from petsc4py.PETSc import ScalarType  # type: ignore

import matplotlib.pyplot as plt
import numpy as np
import ufl


def f(x):
    """Define a função $(4x^3 - 6x)e^{-x^2}$."""
    return (4*x[0]**3 - 6*x[0]) * (ufl.exp(-x[0]**2))

def initial_conditions(x):
    """Fronteira esquerda (ou condiçoes iniciais) do problema"""
    return np.isclose(x[0], 0.0)

def boundary_conditions(x):
    """Fronteira direita do problema""" 
    return np.isclose(x[0], 1.0)

def u_exact(x):
    return x * np.exp(-x**2)


msh = mesh.create_unit_interval(comm=MPI.COMM_WORLD, nx=10)
V = fem.functionspace(msh, ("Lagrange", 1))
v = ufl.TestFunction(V)
u = ufl.TrialFunction(V)
x = ufl.SpatialCoordinate(msh)

### Apply Dirichlet Boundary Conditions
inital_conditions_facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=initial_conditions
)

boundary_conditions_facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=boundary_conditions
)

dofs_initial = fem.locate_dofs_topological(
    V=V,
    entity_dim=(msh.topology.dim - 1), 
    entities=inital_conditions_facets
)

initial_bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs_initial, V=V)

dofs_bounds = fem.locate_dofs_topological(
    V=V, 
    entity_dim=(msh.topology.dim - 1), 
    entities=boundary_conditions_facets
)

bounds_bc = fem.dirichletbc(value=np.exp(-1), dofs=dofs_bounds, V=V)

boundary_conditions = [initial_bc, bounds_bc]


dx = ufl.Measure("cell", domain=msh)
a  = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
L  = ufl.inner(f(x), v) * dx


problem = LinearProblem(
    -a,
    L,
    bcs=boundary_conditions,
    petsc_options_prefix="demo_poisson_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
)


solution = problem.solve()

x_coords = msh.geometry.x[:, 0]
u_values = solution.x.array
 

out_folder = Path("out_poisson")
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

