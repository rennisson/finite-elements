from plot import plot_graphs_3d

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np
import time
import ufl

PI  = ufl.pi
sin = ufl.sin
N = 30 # Mesh size

def f(x):
    """Define a função $(4x^3 - 6x)e^{-x^2}$."""
    return -3 * PI**2 * sin(PI * x[0]) * sin(PI * x[1]) * sin(PI * x[2])

def u_exact(x):
    x, y, z = x
    return sin(PI * x) * sin(PI * y) * sin(PI * z)

start = time.perf_counter()
msh = mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), 
    n=(N, N, N),
    cell_type=mesh.CellType.tetrahedron
)
end = time.perf_counter()
print(f"Criação do mesh grid: {end - start:0.6f} segundos")

V = fem.functionspace(msh, ("Lagrange", 1)) 
v = ufl.TestFunction(V)
u = ufl.TrialFunction(V)
x = ufl.SpatialCoordinate(msh)

start = time.perf_counter()
# Dirichlet Boundary Condition
facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.ones(x.shape[1], dtype=bool)  # Finds all boundary facets
)

dofs_boundary = fem.locate_dofs_topological(
    V=V,
    entity_dim=(msh.topology.dim - 1), 
    entities=facets
)

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs_boundary, V=V)

boundary_conditions = [bc]
end = time.perf_counter()
print(f"Definindo e aplicando condições de contorno: {end - start:0.6f} segundos")


start = time.perf_counter()
dx = ufl.Measure("cell", domain=msh)
a  = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
L  = ufl.inner(f(x), v) * dx


problem = LinearProblem(
    -a,
    L,
    bcs=boundary_conditions,
    petsc_options_prefix="demo_poisson_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True}
)


solution = problem.solve()

end = time.perf_counter()
print(f"Resolvendo o sistema: {end - start:0.6f} segundos")

plot_graphs_3d(mesh_domain=msh, u_exact=u_exact, solution=solution)
