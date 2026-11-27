# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
from math import factorial
from typing import TYPE_CHECKING
from dataclasses import dataclass

from madng_tpsa import symplectic_unit_matrix, TpsaMap, TpsaTensor

from .longitudinal import MapLongitudinalIndices
from .helpers import as_tpsa_tensor, symplectic_inverse, exp_nilpotent_matrix, matrix_to_map
from ..linear_normal_form import get_linear_normal_form

if TYPE_CHECKING:
    from madng_tpsa import Tpsa
    from .longitudinal import Method


@dataclass(slots=True, frozen=True, eq=False)
class LinearNormalisation:
    method: Method
    normalising_matrix: TpsaTensor          # Param-dependent transverse/full matrix
    inverse_normalising_matrix: TpsaTensor
    normalised_matrix: TpsaTensor           # Param-dependent rotation family
    normalising_map: TpsaMap                # Canonical map Cartesian -> centred physical
    inverse_normalising_map: TpsaMap
    normalised_map: TpsaMap                 # Full linearly normalised map


def linear_normalise(
    centred_map: TpsaMap,
    *,
    method: Method | None = None,
    max_parameter_degree: int | None = None,
    tol: float = 1e-11,
    _lidx: MapLongitudinalIndices | None = None,
) -> LinearNormalisation:
    """Parameter-dependent real linear normal form.

    All Taylor coefficients of A are computed by solving the parameter-wise
    homological equations, not by finite-differencing eigenvectors. In 5D/56D
    the parameter family includes pzeta; in 4D/6D it contains external knobs.
    """
    if _lidx is None:
        _lidx = MapLongitudinalIndices.from_method(method)
        _lidx.check_map(centred_map)
    elif method is not None:
        raise ValueError("Cannot provide both `method` and `_lidx`!")
    d = centred_map.descriptor
    rk = _lidx.rank
    pi = _lidx.momentum_index

    if not np.allclose(centred_map.const_part[:rk], 0, atol=tol):
        raise ValueError('Map not centered!')
    # TODO: I would additionally test the entire parameter-dependent fixed-point residual, rather than just its constant coefficient.

    # print("Jacobian family centered map:")
    family = _jacobian_family(centred_map, _lidx)
    # check_family_symplecticity(family)
    M0 = family.const_part[:rk, :rk]
    a0, ai0, r0 = _normalisation_at_reference(M0, rk)
    A = as_tpsa_tensor(a0, d)
    Ainv = as_tpsa_tensor(ai0, d)
    B = Ainv @ family @ A
    R = as_tpsa_tensor(r0, d)

    default_degree = d.order - 1
    if pi is None:
        default_degree = min(default_degree, d.param_order if d.num_params else 0)
    degree_limit = default_degree if max_parameter_degree is None else max_parameter_degree
    if not 0 <= degree_limit <= default_degree:
        raise ValueError(f'max_parameter_degree must lie in [0, {default_degree}]')

    solver = HomologicalSolver(r0)

    for deg in range(1, degree_limit + 1):
        # Compare with the *already accumulated* parameter-dependent rotation.
        # This matters beyond first order: the raw second-order coefficient of
        # exp(mu(k)*S) is not itself an infinitesimal Hamiltonian matrix.

        if deg == 2:
            previous = B @ R.T
            first_order = coefficient_matrices(previous, 1)
            print('Remaining first-order perturbations:')
            for monomial, matrix in first_order.items():
                print(monomial, np.abs(matrix).max())

        relative = B @ R.T  # R^-1 = R^T
        corrections = coefficient_matrices(relative, deg)
        K = as_tpsa_tensor(np.zeros((rk, rk)), d)
        H = as_tpsa_tensor(np.zeros((rk, rk)), d)
        # for mono, q in corrections.items():
        #     kk, hh = solver.solve(q)
        for mono, q in corrections.items():
            try:
                kk, hh = solver.solve(q)
            except ValueError:
                S = solver.S
                defect = q.T @ S + S @ q

                defect_scale = (
                    np.abs(q).T @ np.abs(S)
                    + np.abs(S) @ np.abs(q)
                )
                allowed = solver.atol + solver.rtol * defect_scale

                ratio = np.abs(defect) / allowed
                worst = np.unravel_index(np.argmax(ratio), ratio.shape)

                print(f'Degree: {deg}')
                print(f'Monomial: {mono}')
                print(f'Largest |q|: {np.max(np.abs(q)):.6e}')
                print(f'Largest |defect|: {np.max(np.abs(defect)):.6e}')
                print(f'Worst violation: {ratio[worst]:.6e}')
                print(f'Worst entry: {worst}')
                print(f'q =\n{q}')
                raise
            for i in range(rk):
                for j in range(rk):
                    if kk[i, j] != 0:
                        K[i, j].set(mono, float(kk[i, j]))
                    if hh[i, j] != 0:
                        H[i, j].set(mono, float(hh[i, j]))
        if any(not t.is_zero() for t in K.flat):
            expK = exp_nilpotent_matrix(K, degree_limit, deg)
            expMinusK = exp_nilpotent_matrix(-K, degree_limit, deg)
            A = A @ expK
            B = expMinusK @ B @ expK
        if any(not t.is_zero() for t in H.flat):
            expH = exp_nilpotent_matrix(H, degree_limit, deg)
            R = expH @ R

    ac, aci, Ainv = _linear_coordinate_maps(A, _lidx)
    normalised = aci << centred_map << ac
    return LinearNormalisation(_lidx.method, A, Ainv, R, ac, aci, normalised)

def _jacobian_family(m: TpsaMap, lidx: MapLongitudinalIndices) -> TpsaTensor:
    """Transverse Jacobian at zero transverse amplitude, as a polynomial
    in external knobs and (for coasting) conserved momentum.
    """
    pi = lidx.momentum_index
    nv = m.num_vars
    d = m.descriptor
    # Take all derivatives in one call, then project onto the parameter
    # subspace. For coasting beams pzeta remains a TPSA variable; all
    # oscillatory coordinates (and zeta, in 56D) are set to zero. External
    # descriptor parameters are preserved automatically by compose().
    projection = TpsaMap([
        d.var(i + 1) if i == pi else d.zero()
        for i in range(nv)
    ])
    return m.jacobian().compose(projection)

def check_family_symplecticity(family):
    d = family.descriptor
    n = family.shape[0]
    S = as_tpsa_tensor(symplectic_unit_matrix(n), d)
    defect = family.T @ S @ family - S
    print('Symplecticity defect by parameter degree:')
    for deg in range(d.order):
        coefficients = [
            abs(v)
            for t in defect.flat
            for v in t.homogeneous(deg).monomial_coeffs(tol=0).values()
        ]
        err = max(coefficients, default=0.0)
        print(f'  Degree {deg}: {err:.6e}')

def _normalisation_at_reference(m0: np.ndarray, rk: int):
    if rk == 4:
        # Xtrack's eigenvector/phase convention, using its documented
        # dummy longitudinal plane for an otherwise 4D normal form.
        full = np.eye(6)
        full[:4, :4] = m0
        a, ai, r, _ = get_linear_normal_form(full, only_4d_block=True)
        a, ai, r = a[:4, :4], ai[:4, :4], r[:4, :4]
    else:
        a, ai, r, _ = get_linear_normal_form(m0)

    # Verify
    ss = symplectic_unit_matrix(rk)
    if not np.allclose(a.T @ ss @ a, ss, atol=1e-8, rtol=0):
        raise ValueError('Nominal Xtrack eigenvectors are not symplectically normalised')
    if not np.allclose(ai @ m0 @ a, r, atol=1e-8, rtol=0):
        raise ValueError('Nominal matrix is not reproduced by its normal form')
    return a, ai, r

def coefficient_matrices(matrix: TpsaTensor, degree: int):
    n = matrix.shape[0]
    m = matrix.shape[1]
    result = {}
    for i in range(n):
        for j in range(m):
            part = matrix[i, j].homogeneous(degree)
            for monomial, value in part.monomial_coeffs(tol=0).items():
                q = result.setdefault(monomial, np.zeros((n, n)))
                q[i, j] = value
    return result

def _linear_coordinate_maps(A: TpsaTensor, lidx: MapLongitudinalIndices):
    Ainv = symplectic_inverse(A)
    forward = matrix_to_map(A, lidx)
    backward = matrix_to_map(Ainv, lidx)
    return forward, backward, Ainv


# F. The parameter-dependent homological equations are correct
# This is the most interesting mathematical part of the implementation, and I agree with your approach.

# At a given homogeneous parameter order \(n\), you seek a Hamiltonian matrix \(K_n\) such that
# \[
# B_{\mathrm{new}}
# =e^{-K_n}Be^{K_n}.
# \]

# Writing the current matrix as
# \[
# B=(I+Q_n)R+\mathcal O(k^{n+1}),
# \]

# one obtains
# \[
# B_{\mathrm{new}}R^{-1}
# =
# I+Q_n+RK_nR^{-1}-K_n
# +\mathcal O(k^{n+1}).
# \]

# Therefore, requiring the remaining perturbation to be rotational gives
# \[
# \boxed{
# Q_n+\operatorname{Ad}_R(K_n)-K_n=H_n
# }
# \]

# which is exactly the equation implemented in _solve_homological_matrix().

# Your approach of keeping the accumulated rotation matrix and computing the relative perturbation,
# relative = _matmul(bmat, inverse_r)



# is particularly important at higher orders. Without it, the higher-order corrections would not generally belong to the symplectic Lie algebra. linear.py


# The canonical longitudinal completions of both the orbit translation and the normalising matrix also have the correct signs, given your symplectic-matrix convention. I would retain them.

def _hamiltonian_basis(n: int) -> list[np.ndarray]:
    ss = symplectic_unit_matrix(n)
    out = []
    for i in range(n):
        for j in range(i, n):
            h = np.zeros((n, n))
            h[i, j] = 1
            h[j, i] = 1
            out.append(ss @ h)
    return out

class HomologicalSolver:
    def __init__(
        self,
        r0: np.ndarray,
        *,
        rcond: float = 1e-11,
        atol: float = 1e-11,
        rtol: float = 1e-9
    ):
        self.r0 = np.asarray(r0, dtype=float)
        self.n = len(r0)
        self.S = symplectic_unit_matrix(self.n)
        self.basis = np.asarray(_hamiltonian_basis(self.n))

        rinv = self.r0.T  # Valid for an orthogonal rotation.
        operator = np.column_stack([(self.r0 @ k @ rinv - k).ravel() for k in self.basis])

        # Remove the rotational gauge freedom.
        gauge = np.array([
            [(k[i, i+1] - k[i+1, i]) / 2 for k in self.basis]
            for i in range(0, self.n, 2)
        ])
        self.lhs = np.vstack([operator, gauge])

        # Factorise ONCE.
        u, singular_values, vh = np.linalg.svd(self.lhs, full_matrices=False)
        self.singular_values = singular_values
        if singular_values[-1] <= rcond * singular_values[0]:
            raise ValueError(
                "Homological operator is singular or "
                "ill-conditioned; possible resonance. "
                f"Relative smallest singular value: "
                f"{singular_values[-1] / singular_values[0]:.3e}"
            )
        self.condition_number = (singular_values[0] / singular_values[-1])
        self.pinv = ((vh.T / singular_values) @ u.T)
        self.atol = atol
        self.rtol = rtol

    def rotation_part(self, q):
        h = np.zeros_like(q)
        for i in range(0, self.n, 2):
            angle = (q[i, i+1] - q[i+1, i]) / 2
            h[i:i+2, i:i+2] = (angle * self.S[i:i+2, i:i+2])
        return h

    def solve(self, q):
        q = np.asarray(q, dtype=float)

        # Q must belong to the Hamiltonian Lie algebra.
        defect = q.T @ self.S + self.S @ q
        defect_scale = (np.abs(q).T @ np.abs(self.S) + np.abs(self.S) @ np.abs(q))
        if np.any(np.abs(defect) > self.atol + self.rtol * defect_scale):
            raise ValueError("Input perturbation is not Hamiltonian")
        h = self.rotation_part(q)
        rhs = np.concatenate([(h - q).ravel(), np.zeros(self.n // 2)])
        weights = self.pinv @ rhs
        k = np.einsum("a,aij->ij", weights, self.basis)
        residual = (q + self.r0 @ k @ self.r0.T - k - h)
        scale = max(1.0, np.max(np.abs(q)), np.max(np.abs(h)), np.max(np.abs(k)))
        if np.max(np.abs(residual)) > (self.atol + self.rtol * scale):
            raise ValueError(
                "Homological equation not solved "
                "to the requested tolerance"
            )
        return k, h
