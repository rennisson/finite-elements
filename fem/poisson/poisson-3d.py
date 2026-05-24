from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from pathlib import Path
from petsc4py.PETSc import ScalarType  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import time
import ufl

PI  = ufl.pi
sin = ufl.sin
N = 50 # Mesh size

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

out_folder = Path("results")
out_folder.mkdir(parents=True, exist_ok=True)

# Extrair coordenadas e valores
x_coords = msh.geometry.x[:, 0]
y_coords = msh.geometry.x[:, 1]
z_coords = msh.geometry.x[:, 2]

u_values = solution.x.array.real
u_exact_values = np.array([u_exact(msh.geometry.x[i]) for i in range(len(msh.geometry.x))])

# Calcular min e max global
vmin = min(u_values.min(), u_exact_values.min())
vmax = max(u_values.max(), u_exact_values.max())

# Fatias em z = 0.5
tolerance = 0.05
mask_z = np.isclose(z_coords, 0.5, atol=tolerance)

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
ax[0].set_title(f'Numerical solution (z=0.5) {N=}')
ax[0].set_aspect('equal')

# Solução analítica
scatter2 = ax[1].scatter(x_slice, y_slice, c=u_exact_slice, cmap='viridis', s=50, vmin=vmin, vmax=vmax)
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title('Analytical solution (z=0.5)')
ax[1].set_aspect('equal')

# Colorbar
cbar = plt.colorbar(scatter2, ax=ax, label='u(x, y, 0.5)', shrink=0.8)
plt.savefig(out_folder / f"poisson_3d_solution_slice_{N}points.png", dpi=150)
plt.close()

print(f"Erro máximo na fatia z=0.5: {np.max(np.abs(u_slice - u_exact_slice)):.2e}")
