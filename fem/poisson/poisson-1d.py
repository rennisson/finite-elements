from plot import plot_graphs_1d

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

def u_exact(x):
    return x * np.exp(-x**2)

def initial_conditions(x):
    """Fronteira esquerda (ou condiçoes iniciais) do problema"""
    return np.isclose(x[0], 0.0)

def boundary_conditions(x):
    """Fronteira direita do problema""" 
    return np.isclose(x[0], 1.0)


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

plot_graphs_1d(mesh_domain=msh, u_exact=u_exact, solution=solution)
