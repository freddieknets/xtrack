# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from madng_tpsa import TpsaMap, symplectic_unit_matrix


def max_coefficient(tensor):
    return max(
        (abs(c) for t in tensor.flat for c in t.monomial_coeffs(tol=0.).values()),
        default=0.,
    )

def inspect_residual(name, residual):
    print(name)
    for degree in range(residual.truncation_order + 1):
        error = max_coefficient(residual.homogeneous(degree))
        print(f"  Degree {degree}: {error:.6e}")

def inspect_symplectic_defect(m, lidx):
    E = lidx.symplectic_defect(m)
    E_anti = (E - E.T) / 2
    E_sym = (E + E.T) / 2

    print(f"{'Degree':>6} {'Antisymmetric':>16} {'Symmetric':>16}")
    for n in range(m.truncation_order):
        anti = max_coefficient(E_anti.homogeneous(n))
        sym = max_coefficient(E_sym.homogeneous(n))
        print(f"{n:6d} {anti:16.6e} {sym:16.6e}")

def linear_matrix_to_map(matrix, descriptor):
    variables = descriptor.vars()
    return TpsaMap([
        sum(float(matrix[i, j]) * variables[j] for j in range(descriptor.num_vars)) 
        for i in range(descriptor.num_vars)
    ])

def inspect_linear_family(m, lidx):
    from .linear import _jacobian_family
    family = _jacobian_family(m, lidx)
    rank = lidx.rank
    family = family[:rank, :rank]
    S = symplectic_unit_matrix(rank)
    E = (family.T @ S @ family - S).truncate(m.truncation_order - 1)
    inspect_residual("Linear family defect", E)

def inspect_inverse(m, mi):
    identity = TpsaMap(m.descriptor.vars())
    left = (m << mi) - identity
    right = (mi << m) - identity

    print(
        f"{'Degree':>6}"
        f"{'Map':>14}"
        f"{'Inverse':>14}"
        f"{'Left error':>14}"
        f"{'Right error':>14}"
    )

    for n in range(m.truncation_order + 1):
        values = [max_coefficient(t.homogeneous(n)) for t in (m, mi, left, right)]
        print(f"{n:6d}" + "".join(f"{v:14.5e}" for v in values))
