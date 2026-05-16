from utils.fem_plots import plot_basis_functions

import basix
import basix.ufl
import numpy as np

# Create an element
element = basix.ufl.element(
    family=basix.ElementFamily.P, # Polynomials
    cell=basix.CellType.triangle,
    degree=1,
    lagrange_variant=basix.LagrangeVariant.equispaced,
    dtype=np.float64,
)

# Lattice of points in a triangle
lattice = basix.create_lattice(
    basix.CellType.triangle, 
    25, 
    basix.LatticeType.equispaced, 
    True
)

values = element.tabulate(0, lattice)[0, :, :]
plot_basis_functions(element=element, lattice=lattice, basis_functions_values=values)