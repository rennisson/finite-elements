import numpy as np
import ufl

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
from pathlib import Path
from plot import plot_graphs_1d

def f(x):
    """Define a função (4x^3 - 6x)e^{-x^2}."""
    return (4*x[0]**3 - 6*x[0]) * (ufl.exp(-x[0]**2))

def u_exact(x):
    return x * np.exp(-x**2)

def left_boundary(x):
    """Fronteira esquerda (ou condições iniciais) do problema"""
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    """Fronteira direita do problema""" 
    return np.isclose(x[0], 1.0)


def solve_poisson_1d(nx: int):
    """
    Resolve a equação de Poisson 1D usando FEniCSx para um dado tamanho de malha nx.
    
    Retorna:
        tuple: (x, y) ordenados da esquerda para a direita.
    """
    print(f"\n--- Resolvendo para malha com nx = {nx} ---")
    
    # Criação da Malha e Espaço de Funções
    msh = mesh.create_unit_interval(comm=MPI.COMM_WORLD, nx=nx)
    V = fem.functionspace(msh, ("Lagrange", 1))
    
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Condições de Contorno de Dirichlet
    facets_left = mesh.locate_entities_boundary(msh, dim=0, marker=left_boundary)
    dofs_left = fem.locate_dofs_topological(V, entity_dim=0, entities=facets_left)
    bc_left = fem.dirichletbc(value=ScalarType(0), dofs=dofs_left, V=V)

    facets_right = mesh.locate_entities_boundary(msh, dim=0, marker=right_boundary)
    dofs_right = fem.locate_dofs_topological(V, entity_dim=0, entities=facets_right)
    bc_right = fem.dirichletbc(value=ScalarType(np.exp(-1.0)), dofs=dofs_right, V=V)

    bcs = [bc_left, bc_right]

    # Formulação Variacional
    dx = ufl.Measure("cell", domain=msh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(f(x), v) * dx

    # Solução do Sistema Linear
    problem = LinearProblem(
        -a, L, bcs=bcs,
        petsc_options_prefix="demo_poisson_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    solution = problem.solve()
    
    # --- EXTRAÇÃO E ORDENAÇÃO DOS DADOS ---
    # Extrai a geometria (os pontos X) do espaço de funções V
    points_x = V.tabulate_dof_coordinates()[:, 0] 
    valores_y = solution.x.array
    
    # Ordena os pontos x e aplica a mesma ordenação aos valores y
    indices_ordenados = np.argsort(points_x)
    x_ordenado = points_x[indices_ordenados]
    y_ordenado = valores_y[indices_ordenados]
    
    plot_graphs_1d(mesh_domain=msh, u_exact=u_exact, solution=solution)

    # Reshape para matrizes coluna (N, 1)
    return x_ordenado.reshape(-1, 1), y_ordenado.reshape(-1, 1)

def main():
    # Cria o diretório de destino caso não exista
    output_dir = Path("npz_fem")
    output_dir.mkdir(exist_ok=True)
    
    mesh_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    for nx in mesh_sizes:
        # Resolve o problema para o tamanho atual
        x, y = solve_poisson_1d(nx)
        
        # Salva o resultado no formato .npz. 
        nome_arquivo = output_dir / f'dados_fem_nx{nx}.npz'
        np.savez_compressed(
            nome_arquivo, 
            x_fem=x, 
            y_fem=y
        )
        
        print(f"Salvo com sucesso: {nome_arquivo}")

if __name__ == "__main__":
    main()
